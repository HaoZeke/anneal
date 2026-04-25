//! The move-kernel trait of the IISE manuscript: `Move : S x R_>0 -> Delta(S)`.

use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use rand::Rng;

use crate::neigh::Neighborhood;

/// A temperature-indexed proposal kernel.
///
/// IISE manuscript law L2 requires `supp(Move(i, T)) subseteq Neigh(i)`.
/// Implementors override `supports_in` to advertise this constraint per
/// neighborhood. Default `true` because every shipped move kernel
/// (Gaussian, Cauchy, Tsallis-q_v) has full-R^n support, which subsumes
/// every shipped neighborhood.
pub trait MoveKernel<T: Float>: Send + Sync {
    /// Draws a proposal point from the kernel at temperature `t`.
    fn propose<R: Rng>(&self, i: ArrayView1<T>, t: T, rng: &mut R) -> Array1<T>;

    /// Witnesses L2: returns `true` iff `supp(propose) subseteq n`. Default `true`.
    fn supports_in<N: Neighborhood<T>>(&self, _n: &N) -> bool {
        true
    }
}
