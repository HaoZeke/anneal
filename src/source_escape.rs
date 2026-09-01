//! Budgeted perturb--quench source generation for hybrid PES exploration.
//!
//! A proposal can come from a rigid molecular move, an MD endpoint, a graph
//! move, or an arbitrary-dimensional optimizer. This layer owns only the
//! rgmin quench and its potential-call boundary. Exact basin identity remains
//! the catalog's decision, so source generation and transition discovery can
//! share evidence without conflating their search coordinates or their PES.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array1, ArrayView1};

use crate::pes_exploration::{PesSurface, QuenchedMinimum, quench_minimum_with_norm};

/// Numerical and evaluation-budget contract for one source escape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceEscapeConfig {
    /// Largest number of PES evaluations the quench may issue.
    pub maximum_evaluations: u64,
    /// rgmin L-BFGS iteration limit.
    pub quench_steps: usize,
    /// Componentwise gradient tolerance supplied to rgmin.
    pub gradient_tolerance: f64,
    /// Independent Euclidean gradient-norm certification tolerance.
    pub gradient_norm_tolerance: f64,
}

/// One force-certified minimum produced by an escape proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceEscapeRecord {
    /// Certified minimum, including the fresh gradient evidence.
    pub minimum: QuenchedMinimum,
    /// PES evaluations actually issued by this escape.
    pub charged_evaluations: u64,
}

/// A failed escape whose consumed PES work remains observable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceEscapeFailure {
    /// Stable diagnostic from the quench or surface boundary.
    pub error: String,
    /// PES evaluations issued before the failure.
    pub charged_evaluations: u64,
}

/// Scientific result of one budgeted perturb--quench escape.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceEscapeOutcome {
    /// The proposal quenched to a force-certified minimum.
    Converged(SourceEscapeRecord),
    /// The quench failed or exhausted its evaluation slice.
    Failed(SourceEscapeFailure),
}

#[derive(Debug)]
enum LimitedSurfaceError {
    Exhausted,
    Surface(String),
}

impl Display for LimitedSurfaceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("source escape evaluation budget exhausted"),
            Self::Surface(error) => formatter.write_str(error),
        }
    }
}

struct LimitedSurface<'a, S: ?Sized> {
    surface: &'a S,
    maximum_evaluations: u64,
    calls: AtomicU64,
}

impl<S: ?Sized> LimitedSurface<'_, S> {
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl<S> PesSurface for LimitedSurface<'_, S>
where
    S: PesSurface + ?Sized,
{
    type Error = LimitedSurfaceError;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |calls| {
                (calls < self.maximum_evaluations).then_some(calls + 1)
            })
            .map_err(|_| LimitedSurfaceError::Exhausted)?;
        self.surface
            .evaluate(coordinates)
            .map_err(|error| LimitedSurfaceError::Surface(error.to_string()))
    }
}

/// Quenches one caller-generated escape proposal under a hard PES-call cap.
///
/// The returned charge counts calls even when the underlying surface rejects
/// one. Calls refused at the cap do not enter the charge. The proposal and the
/// converged point may have any dimension accepted by `surface`; molecular
/// invariance belongs to the proposal kernel, not this generic quench layer.
pub fn quench_source_escape<S>(
    surface: &S,
    proposal: ArrayView1<'_, f64>,
    config: &SourceEscapeConfig,
) -> SourceEscapeOutcome
where
    S: PesSurface + ?Sized,
{
    let limited = LimitedSurface {
        surface,
        maximum_evaluations: config.maximum_evaluations,
        calls: AtomicU64::new(0),
    };
    let result = quench_minimum_with_norm(
        &limited,
        proposal,
        config.quench_steps,
        config.gradient_tolerance,
        config.gradient_norm_tolerance,
    );
    let charged_evaluations = limited.calls();
    match result {
        Ok(minimum) => SourceEscapeOutcome::Converged(SourceEscapeRecord {
            minimum,
            charged_evaluations,
        }),
        Err(error) => SourceEscapeOutcome::Failed(SourceEscapeFailure {
            error: error.to_string(),
            charged_evaluations,
        }),
    }
}
