//! The neighborhood trait of the IISE manuscript: `Neigh : S -> 2^S`.

use eindir_core::Bounds;
use ndarray::ArrayView1;
use num_traits::Float;
use rand_distr::uniform::SampleUniform;

/// A neighborhood relation on the state space.
///
/// IISE manuscript law L1 requires `j in Neigh(i) <=> i in Neigh(j)`.
/// Implementors override `is_symmetric` to advertise this. The conservative
/// default is `false`.
pub trait Neighborhood<T: Float>: Send + Sync {
    /// Returns `true` iff `j` is in the neighborhood of `i`.
    fn contains(&self, i: ArrayView1<T>, j: ArrayView1<T>) -> bool;

    /// Witnesses L1: returns `true` iff the relation is symmetric.
    fn is_symmetric(&self) -> bool {
        false
    }
}

/// Unconstrained neighborhood on `R^dim`: every same-dimension point is a
/// neighbor of every other point.
#[allow(non_camel_case_types)]
#[derive(Clone, Debug)]
pub struct ContinuousR_n {
    /// Ambient dimension `n`.
    pub dim: usize,
}

impl ContinuousR_n {
    /// Constructs an unconstrained neighborhood on `R^dim`.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<T: Float + Send + Sync> Neighborhood<T> for ContinuousR_n {
    fn contains(&self, i: ArrayView1<T>, j: ArrayView1<T>) -> bool {
        i.len() == self.dim && j.len() == self.dim
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}

/// Box-constrained neighborhood: `j in Neigh(i)` iff both `i` and `j` lie in
/// the supplied `Bounds`. Symmetric by construction.
#[derive(Clone, Debug)]
pub struct BoxConstrained<T: Float> {
    /// The hyperrectangle constraining the state space.
    pub bounds: Bounds<T>,
}

impl<T: Float> BoxConstrained<T> {
    /// Constructs a box-constrained neighborhood on the supplied `Bounds`.
    pub fn new(bounds: Bounds<T>) -> Self {
        Self { bounds }
    }
}

impl<T> Neighborhood<T> for BoxConstrained<T>
where
    T: Float + SampleUniform + Send + Sync + 'static,
    Bounds<T>: Send + Sync,
{
    fn contains(&self, i: ArrayView1<T>, j: ArrayView1<T>) -> bool {
        self.bounds.contains(i) && self.bounds.contains(j)
    }

    fn is_symmetric(&self) -> bool {
        true
    }
}
