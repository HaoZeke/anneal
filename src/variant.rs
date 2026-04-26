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

/// Sweep budget for `checked_with_sweep`. `Off` skips the random
/// property sweeps; `Default` runs 256 samples per law (matches the
/// proptest default per-test budget); `Strict` runs 4096.
#[derive(Clone, Copy, Debug)]
pub enum SweepBudget {
    /// No randomised law sweep; behaves identically to `checked`.
    Off,
    /// 256 samples per law sweep (default proptest budget).
    Default,
    /// 4096 samples per law sweep, for strict correctness checks.
    Strict,
    /// Custom sample count.
    Custom(usize),
}

impl SweepBudget {
    fn n_samples(self) -> usize {
        match self {
            SweepBudget::Off => 0,
            SweepBudget::Default => 256,
            SweepBudget::Strict => 4096,
            SweepBudget::Custom(n) => n,
        }
    }
}

impl<T, O, C, N, M, A> SaVariant<T, O, C, N, M, A>
where
    T: Float,
    O: Objective<T>,
    C: Cooling<T> + Cooling<f64>,
    N: Neighborhood<T> + Neighborhood<f64>,
    M: MoveKernel<T>,
    A: AcceptRule<T> + AcceptRule<f64>,
{
    /// Like `checked`, but additionally runs randomised property sweeps
    /// over `accept` (L3 downhill, L4 temp-monotone), `cool` (L4
    /// epoch-monotone), and `neigh` (L1 symmetry) before returning.
    /// `dim` and `bound` parameterise the L1 neighbourhood sweep over
    /// `[-bound, bound]^dim`. `seed` makes the sweep reproducible.
    ///
    /// This is the v0.3.2 fix for the gap where a deliberately-broken
    /// AcceptRule whose `accept_prob` violated L3 still constructed
    /// cleanly because the trait method that returned the law witness
    /// could lie. See design_pass_04_lit_survey.org task A6.
    #[allow(clippy::too_many_arguments)]
    pub fn checked_with_sweep(
        obj: O,
        cool: C,
        neigh: N,
        mover: M,
        accept: A,
        budget: SweepBudget,
        dim: usize,
        bound: f64,
        seed: u64,
    ) -> Result<Self, LawViolation> {
        let n = budget.n_samples();
        if n > 0 {
            crate::laws::sweep_downhill_accepts(&accept, n, seed)?;
            crate::laws::sweep_accept_monotone_in_temp(&accept, n, seed.wrapping_add(1))?;
            crate::laws::sweep_cooling_monotone(&cool, n.min(1000))?;
            crate::laws::sweep_neighborhood_symmetric(&neigh, dim, bound, n, seed.wrapping_add(2))?;
        }
        Self::checked(obj, cool, neigh, mover, accept)
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
