//! Cluster potentials as objectives.
//!
//! Every cluster example in this crate used to define its own energy and
//! gradient as loose functions. That made the potential an implementation
//! detail of a demonstration: not reusable, not tested, and not the potential
//! any consumer of this crate would actually run. A campaign reported against a
//! potential written inside its own example is reporting on something no one
//! else can obtain.
//!
//! These are [`Objective`] and [`Gradient`] implementations instead, so the
//! cluster driver takes them by the same trait it takes anything else, and
//! [`crate::methods::cluster_search`] can be handed a potential from rgpot
//! without knowing the difference.
//!
//! They are reference implementations. rgpot is where potentials belong, and it
//! reaches this crate as an `eindir_objective_t`; what these are for is to be a
//! second implementation of the same functions, so the two can be checked
//! against each other. Two independent implementations agreeing to six decimals
//! says more about both than either says alone.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

/// A pairwise cluster potential over `n` free points in three dimensions.
///
/// The state is a flat `3n` vector, point-major, which is what the cluster
/// driver and the shape machinery both assume.
#[derive(Debug, Clone)]
pub struct PairPotential {
    /// Points in a configuration.
    pub n_points: usize,
    kind: PairKind,
    bounds: Bounds<f64>,
}

/// Which pair form a [`PairPotential`] carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairKind {
    /// `4 (r^-12 - r^-6)`, depth 1 at `r = 2^(1/6)`.
    LennardJones,
    /// `e^{rho (1 - r)} (e^{rho (1 - r)} - 2)`, depth 1 at `r = 1`.
    ///
    /// Doye and Wales' form with `epsilon = 1` and `r_0 = 1`, so `rho` is the
    /// only parameter and sets the range of the force. Their published global
    /// minima are in these units.
    Morse {
        /// Range parameter; larger is shorter ranged.
        rho: f64,
    },
}

impl PairKind {
    /// Separation at the pair minimum, which sets the natural length scale.
    pub fn r_min(&self) -> f64 {
        match self {
            PairKind::LennardJones => 2.0_f64.powf(1.0 / 6.0),
            PairKind::Morse { .. } => 1.0,
        }
    }

    /// Pair energy and `(dV/dr) / r` at squared separation `r2`.
    ///
    /// The derivative is returned divided by `r` because every caller
    /// multiplies it by the separation vector, and dividing once here avoids a
    /// square root in the Lennard-Jones case entirely.
    #[inline]
    pub fn pair(&self, r2: f64) -> (f64, f64) {
        match self {
            PairKind::LennardJones => {
                let inv2 = 1.0 / r2;
                let inv6 = inv2 * inv2 * inv2;
                let inv12 = inv6 * inv6;
                (4.0 * (inv12 - inv6), -24.0 * inv2 * (2.0 * inv12 - inv6))
            }
            PairKind::Morse { rho } => {
                let r = r2.sqrt();
                let a = (rho * (1.0 - r)).exp();
                // V = a^2 - 2a and dV/dr = 2 rho (a - a^2), which vanishes at
                // r = 1 where a = 1.
                (a * (a - 2.0), 2.0 * rho * (a - a * a) / r)
            }
        }
    }

    /// `d2V/dr2` and `(dV/dr) / r` at squared separation `r2`.
    ///
    /// The two numbers a second-derivative block needs and nothing else. For a
    /// pair term the exact Hessian of `V(|x_i - x_j|)` is
    ///
    /// ```text
    /// A = V''(r) u u^T + (V'(r)/r) (I - u u^T),   H = [[A, -A], [-A, A]]
    /// ```
    ///
    /// so the radial curvature acts along the pair and the transverse curvature,
    /// which is what a rigid rotation of the pair costs, is set by the tension
    /// `V'(r)/r` alone. That second piece is exactly what the stretch-only model
    /// operator in [`crate::model_hessian`] leaves out.
    ///
    /// Lennard-Jones: `V'' = r^-2 (624 r^-12 - 168 r^-6)`, which is written in
    /// inverse powers so the routine takes no square root.
    ///
    /// Morse: with `a = e^{rho (1 - r)}`, `V'' = 2 rho^2 (2 a^2 - a)`.
    #[inline]
    pub fn pair_curvature(&self, r2: f64) -> (f64, f64) {
        match self {
            PairKind::LennardJones => {
                let inv2 = 1.0 / r2;
                let inv6 = inv2 * inv2 * inv2;
                let inv12 = inv6 * inv6;
                (
                    inv2 * (624.0 * inv12 - 168.0 * inv6),
                    -24.0 * inv2 * (2.0 * inv12 - inv6),
                )
            }
            PairKind::Morse { rho } => {
                let r = r2.sqrt();
                let a = (rho * (1.0 - r)).exp();
                (
                    2.0 * rho * rho * (2.0 * a * a - a),
                    2.0 * rho * (a - a * a) / r,
                )
            }
        }
    }
}

impl PairPotential {
    /// A potential over `n_points` points, in a box of half-width `extent`.
    ///
    /// The bounds are the search domain the objective declares, not a physical
    /// container: a cluster is unbounded, and what keeps it together is the
    /// potential. They are set wide enough that a compact structure never
    /// touches them.
    pub fn new(n_points: usize, kind: PairKind, extent: f64) -> Self {
        assert!(n_points >= 2, "a pair potential needs at least two points");
        let dim = 3 * n_points;
        Self {
            n_points,
            kind,
            bounds: Bounds::new(
                Array1::from_elem(dim, -extent),
                Array1::from_elem(dim, extent),
                0.0,
            ),
        }
    }

    /// Lennard-Jones over `n_points`, with a domain scaled to the size.
    pub fn lennard_jones(n_points: usize) -> Self {
        Self::new(
            n_points,
            PairKind::LennardJones,
            Self::extent_for(n_points, PairKind::LennardJones),
        )
    }

    /// Morse over `n_points` at range `rho`.
    pub fn morse(n_points: usize, rho: f64) -> Self {
        let kind = PairKind::Morse { rho };
        Self::new(n_points, kind, Self::extent_for(n_points, kind))
    }

    /// A domain that comfortably holds a compact cluster of this size.
    ///
    /// A close-packed cluster of `n` points has a radius of order
    /// `n^(1/3)` times the pair separation; four times that is loose enough
    /// that the bound is never the thing shaping the answer.
    fn extent_for(n_points: usize, kind: PairKind) -> f64 {
        4.0 * kind.r_min() * (n_points as f64).cbrt()
    }

    /// The pair form.
    pub fn kind(&self) -> PairKind {
        self.kind
    }

    /// Energy and gradient in one pass.
    ///
    /// Fused because a cluster search spends its budget here: the pair loop is
    /// the same for both, so computing them separately doubles the cost of
    /// every relaxation step.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_points;
        let mut e = 0.0;
        let mut g = Array1::zeros(x.len());
        if let Some(coordinates) = x.as_slice() {
            let gradient = g.as_slice_mut().expect("owned gradient is contiguous");
            for i in 0..n {
                let base_i = 3 * i;
                let mut gradient_i = [gradient[base_i], gradient[base_i + 1], gradient[base_i + 2]];
                for j in (i + 1)..n {
                    let base_j = 3 * j;
                    let d = [
                        coordinates[base_i] - coordinates[base_j],
                        coordinates[base_i + 1] - coordinates[base_j + 1],
                        coordinates[base_i + 2] - coordinates[base_j + 2],
                    ];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 <= 0.0 {
                        continue;
                    }
                    let (v, coef) = self.kind.pair(r2);
                    e += v;
                    for k in 0..3 {
                        gradient_i[k] += coef * d[k];
                        gradient[base_j + k] -= coef * d[k];
                    }
                }
                gradient[base_i..base_i + 3].copy_from_slice(&gradient_i);
            }
            return (e, g);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= 0.0 {
                    continue;
                }
                let (v, coef) = self.kind.pair(r2);
                e += v;
                for k in 0..3 {
                    g[3 * i + k] += coef * d[k];
                    g[3 * j + k] -= coef * d[k];
                }
            }
        }
        (e, g)
    }

    /// Energy and gradient, reusing pair terms between atoms marked held.
    ///
    /// A held atom is one the caller pins. Its pairs with other held atoms
    /// keep the value from the first time those coordinates appear, and that
    /// value is added in the same `i < j` place as a fresh pair. The gradient
    /// of a held atom is left at zero: the caller discards it. Every other
    /// pair is the ordinary term, in the ordinary order.
    pub fn value_and_gradient_held(
        &self,
        x: ArrayView1<f64>,
        cache: &mut HeldPairCache,
    ) -> (f64, Array1<f64>) {
        let n = self.n_points;
        debug_assert_eq!(cache.n, n);
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        let Some(coordinates) = x.as_slice() else {
            return self.value_and_gradient(x);
        };
        if let Some(answer) = cache.recall_answer(coordinates) {
            return answer;
        }
        cache.refresh(self.kind, coordinates);
        let g = gradient
            .as_slice_mut()
            .expect("owned gradient is contiguous");
        if let Some(h) = held_prefix(&cache.held) {
            // Held atoms are a prefix, so each row is the cached held partners
            // and then the mobile partners, which is the `i < j` order. A held
            // atom's own gradient is left at zero: the caller discards it.
            for i in 0..h {
                let base_i = 3 * i;
                for j in (i + 1)..h {
                    if let Some(value) = cache.value_at(i, j) {
                        energy += value;
                    }
                }
                for j in h..n {
                    let base_j = 3 * j;
                    let d = [
                        coordinates[base_i] - coordinates[base_j],
                        coordinates[base_i + 1] - coordinates[base_j + 1],
                        coordinates[base_i + 2] - coordinates[base_j + 2],
                    ];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 <= 0.0 {
                        continue;
                    }
                    let (value, coefficient) = self.kind.pair(r2);
                    energy += value;
                    for k in 0..3 {
                        g[base_j + k] -= coefficient * d[k];
                    }
                }
            }
            for i in h..n {
                let base_i = 3 * i;
                let mut gradient_i = [g[base_i], g[base_i + 1], g[base_i + 2]];
                for j in (i + 1)..n {
                    let base_j = 3 * j;
                    let d = [
                        coordinates[base_i] - coordinates[base_j],
                        coordinates[base_i + 1] - coordinates[base_j + 1],
                        coordinates[base_i + 2] - coordinates[base_j + 2],
                    ];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 <= 0.0 {
                        continue;
                    }
                    let (value, coefficient) = self.kind.pair(r2);
                    energy += value;
                    for k in 0..3 {
                        let delta = coefficient * d[k];
                        gradient_i[k] += delta;
                        g[base_j + k] -= delta;
                    }
                }
                g[base_i..base_i + 3].copy_from_slice(&gradient_i);
            }
            cache.remember_answer(coordinates, energy, gradient.as_slice().unwrap());
            return (energy, gradient);
        }
        for i in 0..n {
            let base_i = 3 * i;
            let mut gradient_i = [g[base_i], g[base_i + 1], g[base_i + 2]];
            for j in (i + 1)..n {
                let held_pair = cache.held[i] && cache.held[j];
                if held_pair {
                    if let Some(value) = cache.value_at(i, j) {
                        energy += value;
                    }
                    continue;
                }
                let base_j = 3 * j;
                let d = [
                    coordinates[base_i] - coordinates[base_j],
                    coordinates[base_i + 1] - coordinates[base_j + 1],
                    coordinates[base_i + 2] - coordinates[base_j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= 0.0 {
                    continue;
                }
                let (value, coefficient) = self.kind.pair(r2);
                energy += value;
                for k in 0..3 {
                    let delta = coefficient * d[k];
                    if !cache.held[i] {
                        gradient_i[k] += delta;
                    }
                    g[base_j + k] -= delta;
                }
            }
            if !cache.held[i] {
                g[base_i..base_i + 3].copy_from_slice(&gradient_i);
            }
        }
        cache.remember_answer(coordinates, energy, gradient.as_slice().unwrap());
        (energy, gradient)
    }
}

/// `held` is `true` for a leading run and then stays false.
fn held_prefix(held: &[bool]) -> Option<usize> {
    let mut h = 0;
    while h < held.len() && held[h] {
        h += 1;
    }
    if held[h..].iter().any(|flag| *flag) {
        None
    } else {
        Some(h)
    }
}

/// Pair values between atoms the caller holds fixed.
pub struct HeldPairCache {
    n: usize,
    held: Vec<bool>,
    bits: Vec<u64>,
    /// Upper triangle, index `i * n + j` for `j > i`, both held.
    values: Vec<f64>,
    /// False when the pair is inside the cutoff of a zero separation and the
    /// scalar loop would not add it.
    live: Vec<bool>,
    fresh: bool,
    /// Last coordinates this cache answered, and that answer.
    answer_bits: Vec<u64>,
    answer_energy: f64,
    answer_gradient: Vec<f64>,
}

impl HeldPairCache {
    /// `held[i]` is true when atom `i` does not move.
    pub fn new(n: usize, held: &[bool]) -> Self {
        let mut mask = vec![false; n];
        for (i, flag) in held.iter().take(n).enumerate() {
            mask[i] = *flag;
        }
        Self {
            n,
            held: mask,
            bits: vec![0; 3 * n],
            values: vec![0.0; n * n],
            live: vec![false; n * n],
            fresh: false,
            answer_bits: Vec::new(),
            answer_energy: 0.0,
            answer_gradient: Vec::new(),
        }
    }

    fn recall_answer(&self, coordinates: &[f64]) -> Option<(f64, Array1<f64>)> {
        if self.answer_bits.len() != coordinates.len() {
            return None;
        }
        if self
            .answer_bits
            .iter()
            .zip(coordinates.iter())
            .any(|(bits, value)| *bits != value.to_bits())
        {
            return None;
        }
        Some((
            self.answer_energy,
            Array1::from(self.answer_gradient.clone()),
        ))
    }

    fn remember_answer(&mut self, coordinates: &[f64], energy: f64, gradient: &[f64]) {
        self.answer_bits.clear();
        self.answer_bits
            .extend(coordinates.iter().map(|value| value.to_bits()));
        self.answer_energy = energy;
        self.answer_gradient.clear();
        self.answer_gradient.extend_from_slice(gradient);
    }

    fn value_at(&self, i: usize, j: usize) -> Option<f64> {
        let index = i * self.n + j;
        self.live[index].then_some(self.values[index])
    }

    fn refresh(&mut self, kind: PairKind, coordinates: &[f64]) {
        let n = self.n;
        let mut moved = false;
        if !self.fresh {
            moved = true;
        } else {
            for i in 0..n {
                if !self.held[i] {
                    continue;
                }
                for k in 0..3 {
                    if coordinates[3 * i + k].to_bits() != self.bits[3 * i + k] {
                        moved = true;
                        break;
                    }
                }
                if moved {
                    break;
                }
            }
        }
        if !moved {
            return;
        }
        for i in 0..n {
            if !self.held[i] {
                continue;
            }
            for k in 0..3 {
                self.bits[3 * i + k] = coordinates[3 * i + k].to_bits();
            }
            for j in (i + 1)..n {
                if !self.held[j] {
                    continue;
                }
                let d = [
                    coordinates[3 * i] - coordinates[3 * j],
                    coordinates[3 * i + 1] - coordinates[3 * j + 1],
                    coordinates[3 * i + 2] - coordinates[3 * j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let index = i * n + j;
                if r2 <= 0.0 {
                    self.live[index] = false;
                } else {
                    self.values[index] = kind.pair(r2).0;
                    self.live[index] = true;
                }
            }
        }
        self.fresh = true;
    }
}

impl Objective<f64> for PairPotential {
    fn dim(&self) -> usize {
        3 * self.n_points
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for PairPotential {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        3 * self.n_points
    }
}

impl DifferentiableObjective<f64> for PairPotential {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        PairPotential::value_and_gradient(self, x)
    }
}

mod binary_lj;
mod gmin_tip4p;
mod gupta;
mod tip4p;
pub use binary_lj::BinaryLennardJones;
pub use gmin_tip4p::{
    C6 as GMIN_TIP4P_C6, C12 as GMIN_TIP4P_C12, COULOMB as GMIN_TIP4P_COULOMB, GminTip4p,
    block_from_interleaved_gmin, energy_ohh, gmin_water_geometry, interleaved_gmin_from_block,
    reference_sites as gmin_tip4p_sites, rigid_from_ohh,
};
pub use gupta::{COPPER_Z, GOLD_Z, Gupta};
// The rigid orientation move composes through this name. The helper
// stays on the chart so that move and the potential share one Rodrigues
// product. A target that does not include the move still has to compile.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use tip4p::compose_rotation;
pub use tip4p::{
    COULOMB, EPSILON, HOH_DEG, Q_H, Q_M, R_OH, R_OM, SIGMA, Tip4pCluster, fold_rotation_vectors,
    jorgensen_dimer, random_cluster as random_tip4p_cluster, wales_hodges_minimum,
};

impl crate::pes_exploration::PesSurface for PairPotential {
    type Error = std::convert::Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        Ok(self.value_and_gradient(coordinates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_pairs_match_the_full_sum_after_the_held_gradient_is_cleared() {
        let n = 39;
        let potential = PairPotential::lennard_jones(n);
        let mut held = vec![false; n];
        for i in 0..32 {
            held[i] = true;
        }
        let mut x = Array1::from_shape_fn(3 * n, |index| {
            let point = index / 3;
            let axis = index % 3;
            ((point * 13 + axis * 5) as f64 * 0.17).sin() * 2.4 + point as f64 * 0.05
        });
        let mut cache = HeldPairCache::new(n, &held);
        let (energy, mut gradient) = potential.value_and_gradient_held(x.view(), &mut cache);
        let (full_energy, mut full_gradient) = potential.value_and_gradient(x.view());
        assert_eq!(energy.to_bits(), full_energy.to_bits());
        for i in 0..n {
            if held[i] {
                for k in 0..3 {
                    gradient[3 * i + k] = 0.0;
                    full_gradient[3 * i + k] = 0.0;
                }
            }
        }
        for (got, want) in gradient.iter().zip(full_gradient.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        for i in 32..n {
            x[3 * i] += 0.02 * (i as f64);
        }
        let (energy, mut gradient) = potential.value_and_gradient_held(x.view(), &mut cache);
        let (full_energy, mut full_gradient) = potential.value_and_gradient(x.view());
        assert_eq!(
            energy.to_bits(),
            full_energy.to_bits(),
            "moved mobile atoms"
        );
        for i in 0..n {
            if held[i] {
                for k in 0..3 {
                    gradient[3 * i + k] = 0.0;
                    full_gradient[3 * i + k] = 0.0;
                }
            }
        }
        for (got, want) in gradient.iter().zip(full_gradient.iter()) {
            assert_eq!(got.to_bits(), want.to_bits(), "moved mobile gradient");
        }
    }

    #[test]
    fn contiguous_pair_kernel_matches_the_ordered_reference_bit_for_bit() {
        fn reference(potential: &PairPotential, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let mut energy = 0.0;
            let mut gradient = Array1::zeros(x.len());
            for i in 0..potential.n_points {
                for j in (i + 1)..potential.n_points {
                    let d = [
                        x[3 * i] - x[3 * j],
                        x[3 * i + 1] - x[3 * j + 1],
                        x[3 * i + 2] - x[3 * j + 2],
                    ];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 <= 0.0 {
                        continue;
                    }
                    let (value, coefficient) = potential.kind.pair(r2);
                    energy += value;
                    for k in 0..3 {
                        gradient[3 * i + k] += coefficient * d[k];
                        gradient[3 * j + k] -= coefficient * d[k];
                    }
                }
            }
            (energy, gradient)
        }

        let x = Array1::from_shape_fn(21, |index| {
            let point = index / 3;
            let axis = index % 3;
            ((point * 17 + axis * 11 + 3) as f64).sin() + point as f64 * 0.37
        });
        for potential in [
            PairPotential::lennard_jones(7),
            PairPotential::morse(7, 6.0),
        ] {
            let expected = reference(&potential, x.view());
            let observed = potential.value_and_gradient(x.view());
            assert_eq!(observed.0.to_bits(), expected.0.to_bits());
            for (actual, reference) in observed.1.iter().zip(expected.1.iter()) {
                assert_eq!(actual.to_bits(), reference.to_bits());
            }
        }
    }

    /// Two points at the pair minimum: energy is minus the well depth and the
    /// force vanishes. Fixes depth, position and the sign of the derivative
    /// together, for both forms.
    #[test]
    fn a_pair_at_its_minimum_has_unit_depth_and_no_force() {
        for kind in [PairKind::LennardJones, PairKind::Morse { rho: 6.0 }] {
            let p = PairPotential::new(2, kind, 10.0);
            let r = kind.r_min();
            let x = Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]);
            let (e, g) = p.value_and_gradient(x.view());
            assert!((e + 1.0).abs() < 1e-9, "{kind:?} depth {e}");
            for v in g.iter() {
                assert!(v.abs() < 1e-7, "{kind:?} force {v} at the minimum");
            }
        }
    }

    /// The gradient against a central difference of the energy it
    /// differentiates, which is the part a published energy table cannot check.
    #[test]
    fn the_gradient_matches_a_finite_difference() {
        for kind in [PairKind::LennardJones, PairKind::Morse { rho: 10.0 }] {
            let n = 5;
            let p = PairPotential::new(n, kind, 10.0);
            let mut x = Array1::<f64>::zeros(3 * n);
            for i in 0..n {
                let a = i as f64 * 1.31;
                x[3 * i] = 1.15 * a.cos();
                x[3 * i + 1] = 1.15 * a.sin();
                x[3 * i + 2] = 0.37 * (i % 3) as f64;
            }
            let (_, g) = p.value_and_gradient(x.view());
            let h = 1e-6;
            for k in 0..(3 * n) {
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus[k] += h;
                minus[k] -= h;
                let num = (p.eval(plus.view()) - p.eval(minus.view())) / (2.0 * h);
                assert!(
                    (g[k] - num).abs() < 1e-4 * (1.0 + num.abs()),
                    "{kind:?} coordinate {k}: analytic {} against numeric {num}",
                    g[k]
                );
            }
        }
    }

    /// The range parameter narrows the well without moving it or changing its
    /// depth, which is the property that makes Morse a family rather than one
    /// more potential.
    #[test]
    fn the_morse_range_narrows_the_well_without_moving_it() {
        let soft = PairPotential::morse(2, 3.0);
        let hard = PairPotential::morse(2, 14.0);
        let at_min = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!((soft.eval(at_min.view()) + 1.0).abs() < 1e-9);
        assert!((hard.eval(at_min.view()) + 1.0).abs() < 1e-9);

        let far = Array1::from(vec![0.0, 0.0, 0.0, 1.4, 0.0, 0.0]);
        let e_soft = soft.eval(far.view());
        let e_hard = hard.eval(far.view());
        assert!(e_hard > e_soft, "short range should bind less at 1.4 r0");
        assert!(e_hard.abs() < 0.02, "at rho = 14 the pair is nearly free");
    }

    /// Objective and Gradient have to agree with the fused routine, or a caller
    /// reaching through the traits gets different numbers from one reaching
    /// through the struct.
    #[test]
    fn the_trait_methods_agree_with_the_fused_one() {
        let p = PairPotential::lennard_jones(4);
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.55, 0.95, 0.0, 0.55, 0.32, 0.9,
        ]);
        let (e, g) = p.value_and_gradient(x.view());
        assert_eq!(e, p.eval(x.view()));
        assert_eq!(g, p.grad(x.view()));
        assert_eq!(Objective::dim(&p), 12);
        assert_eq!(Gradient::dim(&p), 12);
    }

    /// A half turn used to come back on the x axis, because the skew part
    /// of the matrix vanishes and the old recovery ignored the axis.
    #[test]
    fn a_half_turn_about_y_keeps_its_axis() {
        let pi = std::f64::consts::PI;
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let composed = compose_rotation(identity, [0.0, pi, 0.0]);
        assert!(composed[0].abs() < 1e-8, "{composed:?}");
        assert!(composed[2].abs() < 1e-8, "{composed:?}");
        assert!(
            (composed[1].abs() - pi).abs() < 1e-8,
            "half turn about y came back as {composed:?}"
        );
    }

    /// The composed vector has to rotate a point the same way as applying
    /// the increment after the original rotation.
    #[test]
    fn composed_rotation_matches_the_matrix_product() {
        let omega = [0.35, -0.2, 0.55];
        let increment = super::tip4p::rotation_matrix([0.4, 0.15, -0.25]);
        let composed = compose_rotation(increment, omega);
        let point = [0.2, -0.7, 1.1];
        let expected = super::tip4p::mat_vec(
            increment,
            super::tip4p::mat_vec(super::tip4p::rotation_matrix(omega), point),
        );
        let got = super::tip4p::mat_vec(super::tip4p::rotation_matrix(composed), point);
        for (left, right) in got.iter().zip(expected.iter()) {
            assert!(
                (left - right).abs() < 1e-9,
                "composed {got:?} against product {expected:?}"
            );
        }
    }

    #[test]
    fn pair_potential_is_the_atomistic_pes_surface() {
        let p = PairPotential::lennard_jones(3);
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.55, 0.95, 0.0]);
        let expected = p.value_and_gradient(x.view());
        let observed = crate::pes_exploration::PesSurface::evaluate(&p, x.view()).unwrap();

        assert_eq!(observed, expected);
    }
}
