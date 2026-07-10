//! Runtime witnesses + the `LawViolation` error returned by
//! `SaVariant::checked` when a tuple of components fails one of the four
//! IISE-manuscript composition laws.
//!
//! `SaVariant::checked_with_sweep` exercises the executable behavior of
//! each law on `n_samples` random inputs rather than trusting Boolean
//! witness methods alone.

use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use thiserror::Error;

use crate::accept::AcceptRule;
use crate::cool::Cooling;
use crate::movekernel::MoveKernel;
use crate::neigh::Neighborhood;

/// A law-violation diagnostic surfaced by `SaVariant::checked` /
/// `SaVariant::checked_with_sweep`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LawViolation {
    /// Sampled validation requires at least one sample per executable law.
    #[error("sampled validation requires a positive sample count")]
    EmptySweep,
    /// L1: `is_symmetric()` returned false on the supplied `Neighborhood`.
    #[error("L1 violation: neighborhood is not symmetric")]
    Symmetry,
    /// L1 (sweep): proptest sampler caught a non-symmetric pair.
    #[error("L1 violation (sweep): contains({i:?}, {j:?}) != contains({j:?}, {i:?})")]
    SymmetrySweep {
        /// First witness point.
        i: Vec<f64>,
        /// Second witness point.
        j: Vec<f64>,
    },
    /// L2: `MoveKernel::supports_in(neigh)` returned false.
    #[error("L2 violation: move support escapes the neighborhood")]
    SupportEscape,
    /// L2 (sweep): a sampled proposal escaped the neighborhood.
    #[error("L2 violation (sweep): proposal from {from:?} at T={temp} produced {to:?}")]
    SupportEscapeSweep {
        /// Feasible state from which the proposal was drawn.
        from: Vec<f64>,
        /// Proposal outside the declared neighborhood.
        to: Vec<f64>,
        /// Positive proposal temperature.
        temp: f64,
    },
    /// The supplied validation cube contained no sampled feasible state.
    #[error("L2 validation domain contained no sampled feasible state")]
    NoFeasibleSweepState,
    /// L3 (sweep): `accept_prob(delta, T) != 1` for some `delta <= 0`.
    #[error("L3 violation (sweep): accept_prob({delta_e}, {temp}) = {p}, expected 1.0")]
    DownhillNotAccepted {
        /// Witness energy delta (`<= 0`).
        delta_e: f64,
        /// Witness temperature.
        temp: f64,
        /// Reported acceptance probability (should be 1).
        p: f64,
    },
    /// L4: `Cooling::is_monotone()` returned false.
    #[error("L4 violation: cooling schedule is not non-increasing")]
    NonMonotoneCooling,
    /// L4 (sweep): cooling schedule caught growing in epoch.
    #[error("L4 violation (sweep): T({k1}) = {t1} < T({k2}) = {t2} (k1 < k2)")]
    NonMonotoneCoolingSweep {
        /// Earlier epoch.
        k1: usize,
        /// Later epoch.
        k2: usize,
        /// Temperature at `k1`.
        t1: f64,
        /// Temperature at `k2` (which exceeded `t1`).
        t2: f64,
    },
    /// L4 (sweep): accept_prob(delta, t1) exceeds accept_prob(delta, t2) with
    /// t1 < t2.
    #[error(
        "L4 violation (sweep): accept_prob({delta_e}, {t1}) = {p1} > accept_prob({delta_e}, {t2}) = {p2}"
    )]
    NonMonotoneAcceptInTemp {
        /// Witness energy delta (`> 0`).
        delta_e: f64,
        /// Lower temperature.
        t1: f64,
        /// Acceptance probability at `t1`.
        p1: f64,
        /// Higher temperature.
        t2: f64,
        /// Acceptance probability at `t2` (should be `>= p1`).
        p2: f64,
    },
}

const SWEEP_TOLERANCE: f64 = 1e-9;

/// Witnesses L3 (downhill always accepts) on an `AcceptRule<f64>` by
/// sampling `n_samples` random `(delta_e <= 0, temp > 0)` pairs and
/// checking `accept_prob == 1.0` for each. Uses `seed` for
/// reproducibility.
pub fn sweep_downhill_accepts<A: AcceptRule<f64>>(
    accept: &A,
    n_samples: usize,
    seed: u64,
) -> Result<(), LawViolation> {
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n_samples {
        let delta_e = -rng.random::<f64>() * 1e3; // delta_e in (-1e3, 0]
        let temp = rng.random::<f64>() * 1e3 + 1e-3; // temp in (1e-3, 1e3]
        let p = accept.accept_prob(delta_e, temp);
        if (p - 1.0).abs() > SWEEP_TOLERANCE {
            return Err(LawViolation::DownhillNotAccepted { delta_e, temp, p });
        }
    }
    Ok(())
}

/// Witnesses L4 by sampling uphill moves at paired temperatures and checking
/// that acceptance probability is non-decreasing in temperature.
pub fn sweep_accept_monotone_in_temp<A: AcceptRule<f64>>(
    accept: &A,
    n_samples: usize,
    seed: u64,
) -> Result<(), LawViolation> {
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n_samples {
        let delta_e = rng.random::<f64>() * 1e2 + 1e-3;
        let t1 = rng.random::<f64>() * 1e2 + 1e-3;
        let t2 = t1 + rng.random::<f64>() * 1e2 + 1e-6;
        let p1 = accept.accept_prob(delta_e, t1);
        let p2 = accept.accept_prob(delta_e, t2);
        if p1 > p2 + SWEEP_TOLERANCE {
            return Err(LawViolation::NonMonotoneAcceptInTemp {
                delta_e,
                t1,
                p1,
                t2,
                p2,
            });
        }
    }
    Ok(())
}

/// Witnesses L4 (cooling non-increasing in epoch) on a `Cooling<f64>` by
/// computing `T(0), T(1), ..., T(n_epochs - 1)` and checking each
/// successive pair.
pub fn sweep_cooling_monotone<C: Cooling<f64>>(
    cool: &C,
    n_epochs: usize,
) -> Result<(), LawViolation> {
    if n_epochs < 2 {
        return Ok(());
    }
    let mut prev = cool.temperature(0);
    for k in 1..n_epochs {
        let t = cool.temperature(k);
        if t > prev + SWEEP_TOLERANCE {
            return Err(LawViolation::NonMonotoneCoolingSweep {
                k1: k - 1,
                k2: k,
                t1: prev,
                t2: t,
            });
        }
        prev = t;
    }
    Ok(())
}

/// Witnesses L1 (neighborhood symmetry) on a
/// `Neighborhood<f64>` by sampling `n_samples` random `(i, j)`
/// position pairs in the supplied bounding box and checking
/// `contains(i, j) == contains(j, i)`.
pub fn sweep_neighborhood_symmetric<N: crate::neigh::Neighborhood<f64>>(
    neigh: &N,
    dim: usize,
    bound: f64,
    n_samples: usize,
    seed: u64,
) -> Result<(), LawViolation> {
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n_samples {
        let i = Array1::from_iter((0..dim).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * bound));
        let j = Array1::from_iter((0..dim).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * bound));
        let ij = neigh.contains(i.view(), j.view());
        let ji = neigh.contains(j.view(), i.view());
        if ij != ji {
            return Err(LawViolation::SymmetrySweep {
                i: i.to_vec(),
                j: j.to_vec(),
            });
        }
    }
    Ok(())
}

/// Witnesses L2 by drawing proposals from sampled feasible states and
/// checking membership before any objective evaluation. The bounding cube is
/// a validation domain supplied by the caller, not a proof over all states.
pub fn sweep_move_support<M, N>(
    mover: &M,
    neigh: &N,
    dim: usize,
    bound: f64,
    n_samples: usize,
    seed: u64,
) -> Result<(), LawViolation>
where
    M: MoveKernel<f64>,
    N: Neighborhood<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut checked = 0usize;
    let max_attempts = n_samples.saturating_mul(100).max(100);
    for _ in 0..max_attempts {
        let from = Array1::from_iter((0..dim).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * bound));
        if !neigh.contains(from.view(), from.view()) {
            continue;
        }
        let temp = rng.random::<f64>() * 1e3 + 1e-3;
        let to = mover.propose(from.view(), temp, &mut rng);
        if !neigh.contains(from.view(), to.view()) {
            return Err(LawViolation::SupportEscapeSweep {
                from: from.to_vec(),
                to: to.to_vec(),
                temp,
            });
        }
        checked += 1;
        if checked == n_samples {
            break;
        }
    }
    if checked == 0 {
        Err(LawViolation::NoFeasibleSweepState)
    } else {
        Ok(())
    }
}
