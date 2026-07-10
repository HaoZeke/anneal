//! `SaVariant<T, O, C, N, M, A>`: a typed tuple of the five IISE-manuscript
//! components plus a `checked` constructor that asserts the four
//! composition laws at instantiation.

use std::marker::PhantomData;

use eindir_core::Objective;
use num_traits::Float;

use crate::accept::{AcceptRule, Metropolis, TsallisAccept};
use crate::cool::{Cooling, LogCool, ReciprocalCool, TsallisCool};
use crate::laws::LawViolation;
use crate::movekernel::{Cauchy, Gaussian, MoveKernel, Reflected, TsallisVisit};
use crate::neigh::{BoxConstrained, ContinuousR_n, Neighborhood};

/// Cooling schedules whose parameter constructors establish L4.
pub trait CertifiedCooling<T: Float>: Cooling<T> {}

impl<T: Float + Send + Sync> CertifiedCooling<T> for LogCool<T> {}
impl<T: Float + Send + Sync> CertifiedCooling<T> for ReciprocalCool<T> {}
impl<T: Float + Send + Sync> CertifiedCooling<T> for TsallisCool<T> {}

/// Neighborhood relations whose implementations establish L1.
pub trait CertifiedNeighborhood<T: Float>: Neighborhood<T> {}

impl<T: Float + Send + Sync> CertifiedNeighborhood<T> for ContinuousR_n {}
impl CertifiedNeighborhood<f64> for BoxConstrained<f64> {}

/// Acceptance rules whose implementations establish L3 and L4.
pub trait CertifiedAcceptance<T: Float>: AcceptRule<T> {}

impl<T: Float + Send + Sync> CertifiedAcceptance<T> for Metropolis {}
impl<T: Float + Send + Sync> CertifiedAcceptance<T> for TsallisAccept<T> {}

/// Move/neighborhood pairs whose support compatibility is certified.
pub trait CertifiedMoveFor<T: Float, N: Neighborhood<T>>: MoveKernel<T> {
    /// Confirms value-level requirements that the type pair cannot encode.
    fn certified_supports(&self, neigh: &N) -> bool;
}

macro_rules! certify_full_space_move {
    ($move:ty) => {
        impl CertifiedMoveFor<f64, ContinuousR_n> for $move {
            fn certified_supports(&self, _neigh: &ContinuousR_n) -> bool {
                true
            }
        }
    };
}

certify_full_space_move!(Gaussian);
certify_full_space_move!(Cauchy);
certify_full_space_move!(TsallisVisit);

impl<M: MoveKernel<f64>> CertifiedMoveFor<f64, BoxConstrained<f64>> for Reflected<M> {
    fn certified_supports(&self, neigh: &BoxConstrained<f64>) -> bool {
        self.bounds.dims == neigh.bounds.dims
            && self.bounds.low == neigh.bounds.low
            && self.bounds.high == neigh.bounds.high
    }
}

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
    /// Constructs a tuple without validating L1-L4.
    ///
    /// This is intended for negative tests and research prototypes. Results
    /// produced through this constructor are excluded from certified-component
    /// claims.
    pub fn unchecked(obj: O, cool: C, neigh: N, mover: M, accept: A) -> Self {
        Self {
            obj,
            cool,
            neigh,
            mover,
            accept,
            _t: PhantomData,
        }
    }

    /// Constructs a variant from component types certified for L1-L4.
    /// Third-party components use [`Self::checked_with_sweep`] instead.
    pub fn checked(obj: O, cool: C, neigh: N, mover: M, accept: A) -> Result<Self, LawViolation>
    where
        C: CertifiedCooling<T>,
        N: CertifiedNeighborhood<T>,
        M: CertifiedMoveFor<T, N>,
        A: CertifiedAcceptance<T>,
    {
        if !cool.is_monotone() {
            return Err(LawViolation::NonMonotoneCooling);
        }
        if !neigh.is_symmetric() {
            return Err(LawViolation::Symmetry);
        }
        if !mover.certified_supports(&neigh) {
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

/// Sweep budget for `checked_with_sweep`. `Default` runs 256 samples per law;
/// `Strict` runs 4096.
#[derive(Clone, Copy, Debug)]
pub enum SweepBudget {
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
    M: MoveKernel<T> + MoveKernel<f64>,
    A: AcceptRule<T> + AcceptRule<f64>,
{
    /// Like `checked`, but additionally runs randomised property sweeps
    /// over `accept` (L3 downhill, L4 temp-monotone), `cool` (L4
    /// epoch-monotone), and `neigh` (L1 symmetry) before returning.
    /// `dim` and `bound` parameterise the L1 neighbourhood sweep over
    /// `[-bound, bound]^dim`. `seed` makes the sweep reproducible.
    ///
    /// The randomized sweep checks the executable law behavior rather than
    /// trusting witness methods alone.
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
        if n == 0 {
            return Err(LawViolation::EmptySweep);
        }
        crate::laws::sweep_downhill_accepts(&accept, n, seed)?;
        crate::laws::sweep_accept_monotone_in_temp(&accept, n, seed.wrapping_add(1))?;
        crate::laws::sweep_cooling_monotone(&cool, n.min(1000))?;
        crate::laws::sweep_neighborhood_symmetric(&neigh, dim, bound, n, seed.wrapping_add(2))?;
        crate::laws::sweep_move_support(&mover, &neigh, dim, bound, n, seed.wrapping_add(3))?;
        if !<C as Cooling<T>>::is_monotone(&cool) {
            return Err(LawViolation::NonMonotoneCooling);
        }
        if !<N as Neighborhood<T>>::is_symmetric(&neigh) {
            return Err(LawViolation::Symmetry);
        }
        if !<M as MoveKernel<T>>::supports_in(&mover, &neigh) {
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
