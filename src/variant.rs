//! `SaVariant<T, O, C, N, M, A>`: a typed tuple of the five IISE-manuscript
//! components plus a `checked` constructor that asserts the four
//! composition laws at instantiation.

use std::marker::PhantomData;

use eindir_core::Objective;
use num_traits::Float;

use crate::accept::{AcceptRule, Metropolis, TsallisAccept};
use crate::cool::{Cooling, LogCool, ReciprocalCool, TsallisCool};
use crate::laws::LawViolation;
use crate::movekernel::{Cauchy, Gaussian, MoveKernel, TsallisVisit};
use crate::neigh::{ContinuousR_n, Neighborhood};

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
    pub fn checked(obj: O, cool: C, neigh: N, mover: M, accept: A) -> Result<Self, LawViolation> {
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

/// Type alias for the Boltzmann (BSA) preset:
/// `(O, LogCool, ContinuousR_n, Gaussian, Metropolis)`.
pub type BoltzmannVariant<O> = SaVariant<f64, O, LogCool<f64>, ContinuousR_n, Gaussian, Metropolis>;

/// Type alias for the Fast (FSA) preset:
/// `(O, ReciprocalCool, ContinuousR_n, Cauchy, Metropolis)`.
pub type FastVariant<O> = SaVariant<f64, O, ReciprocalCool<f64>, ContinuousR_n, Cauchy, Metropolis>;

/// Type alias for the GSA preset:
/// `(O, TsallisCool, ContinuousR_n, TsallisVisit, TsallisAccept)`.
pub type GsaVariant<O> =
    SaVariant<f64, O, TsallisCool<f64>, ContinuousR_n, TsallisVisit, TsallisAccept<f64>>;

/// Constructs the Boltzmann SA variant: logarithmic cooling, isotropic
/// Gaussian moves, Metropolis acceptance, on the unconstrained `R^dim`.
///
/// `dim` is read from `obj.dim()`. `t_init` is the initial temperature;
/// the cooling decays as `T_0 log(2) / log(k+2)` (the `k0 = 2` choice
/// matches the IISE manuscript Section 4 convention). `sigma` is the
/// per-component Gaussian step size.
pub fn boltzmann<O: Objective<f64> + Send + Sync>(
    obj: O,
    t_init: f64,
    sigma: f64,
) -> Result<BoltzmannVariant<O>, LawViolation> {
    let dim = obj.dim();
    SaVariant::checked(
        obj,
        LogCool::new(t_init, 2.0),
        ContinuousR_n::new(dim),
        Gaussian::new(sigma),
        Metropolis,
    )
}

/// Constructs the Fast SA variant: reciprocal cooling, isotropic Cauchy
/// moves, Metropolis acceptance, on the unconstrained `R^dim`.
pub fn fast<O: Objective<f64> + Send + Sync>(
    obj: O,
    t_init: f64,
    gamma: f64,
) -> Result<FastVariant<O>, LawViolation> {
    let dim = obj.dim();
    SaVariant::checked(
        obj,
        ReciprocalCool::new(t_init),
        ContinuousR_n::new(dim),
        Cauchy::new(gamma),
        Metropolis,
    )
}

/// Constructs the GSA variant: Tsallis cooling, Tsallis visit kernel,
/// Tsallis acceptance, on the unconstrained `R^dim`.
///
/// `q_v in (1, 3)` is the visiting index; `q_a` is the acceptance index
/// (`q_a == 1` collapses to Metropolis).
pub fn gsa<O: Objective<f64> + Send + Sync>(
    obj: O,
    t_init: f64,
    q_v: f64,
    q_a: f64,
) -> Result<GsaVariant<O>, LawViolation> {
    let dim = obj.dim();
    SaVariant::checked(
        obj,
        TsallisCool::new(t_init, q_v),
        ContinuousR_n::new(dim),
        TsallisVisit::new(q_v),
        TsallisAccept::new(q_a),
    )
}
