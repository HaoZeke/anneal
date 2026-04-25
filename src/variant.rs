//! `SaVariant<T, O, C, N, M, A>`: a typed tuple of the five IISE-manuscript
//! components plus a `checked` constructor that asserts the four
//! composition laws at instantiation.

use std::marker::PhantomData;

use eindir_core::Objective;
use num_traits::Float;

use crate::accept::AcceptRule;
use crate::cool::Cooling;
use crate::laws::LawViolation;
use crate::movekernel::MoveKernel;
use crate::neigh::Neighborhood;

/// A fully-typed SA variant: an `(Obj, Cool, Neigh, Move, Accept)` tuple
/// satisfying the IISE-manuscript composition laws L1-L4.
pub struct SaVariant<T, O, C, N, M, A>
where
    T: Float,
    O: Objective<T>,
    C: Cooling<T>,
    N: Neighborhood<T>,
    M: MoveKernel<T>,
    A: AcceptRule<T>,
{
    /// The objective.
    pub obj: O,
    /// The cooling schedule.
    pub cool: C,
    /// The neighborhood relation.
    pub neigh: N,
    /// The move kernel.
    pub mover: M,
    /// The acceptance rule.
    pub accept: A,
    _t: PhantomData<T>,
}

impl<T, O, C, N, M, A> SaVariant<T, O, C, N, M, A>
where
    T: Float,
    O: Objective<T>,
    C: Cooling<T>,
    N: Neighborhood<T>,
    M: MoveKernel<T>,
    A: AcceptRule<T>,
{
    /// Constructs a variant after asserting laws L1, L2, L4 via the
    /// runtime witness methods on the supplied components.
    /// L3 (downhill always accepts) is left to the per-impl proptest sweep
    /// since it is a property of the `AcceptRule::accept_prob` function
    /// shape and cannot be captured by a single Boolean witness.
    pub fn checked(
        obj: O,
        cool: C,
        neigh: N,
        mover: M,
        accept: A,
    ) -> Result<Self, LawViolation> {
        if !cool.is_monotone() {
            return Err(LawViolation::NonMonotoneCooling);
        }
        if !neigh.is_symmetric() {
            return Err(LawViolation::Symmetry);
        }
        if !mover.supports_in(&neigh) {
            return Err(LawViolation::SupportEscape);
        }
        Ok(Self {
            obj,
            cool,
            neigh,
            mover,
            accept,
            _t: PhantomData,
        })
    }
}
