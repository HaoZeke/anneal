//! The neighborhood trait of the IISE manuscript: `Neigh : S -> 2^S`.

use ndarray::ArrayView1;
use num_traits::Float;

/// A neighborhood relation on the state space.
///
/// IISE manuscript law L1 requires `j in Neigh(i) <=> i in Neigh(j)`.
/// Implementors override `is_symmetric` to advertise this; the default is
/// `true` because every shipped neighborhood (continuous R^n,
/// box-constrained R^n) satisfies L1 by translation invariance.
pub trait Neighborhood<T: Float>: Send + Sync {
    /// Returns `true` iff `j` is in the neighborhood of `i`.
    fn contains(&self, i: ArrayView1<T>, j: ArrayView1<T>) -> bool;

    /// Witnesses L1: returns `true` iff the relation is symmetric. Default `true`.
    fn is_symmetric(&self) -> bool {
        true
    }
}
