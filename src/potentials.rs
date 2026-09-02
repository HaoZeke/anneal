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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn pair_potential_is_the_atomistic_pes_surface() {
        let p = PairPotential::lennard_jones(3);
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.55, 0.95, 0.0]);
        let expected = p.value_and_gradient(x.view());
        let observed = crate::pes_exploration::PesSurface::evaluate(&p, x.view()).unwrap();

        assert_eq!(observed, expected);
    }
}
