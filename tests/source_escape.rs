use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::pes_exploration::PesSurface;
use anneal_core::source_escape::{SourceEscapeConfig, SourceEscapeOutcome, quench_source_escape};
use ndarray::{Array1, ArrayView1, array};

struct CountingQuadratic {
    calls: AtomicU64,
}

impl CountingQuadratic {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PesSurface for CountingQuadratic {
    type Error = Infallible;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let displacement = coordinates - array![1.0, -2.0, 0.5].view();
        Ok((0.5 * displacement.dot(&displacement), displacement))
    }
}

#[test]
fn generic_source_escape_returns_the_certifying_gradient_and_exact_charge() {
    let surface = CountingQuadratic::new();
    let config = SourceEscapeConfig {
        maximum_evaluations: 100,
        quench_steps: 100,
        gradient_tolerance: 1e-9,
        gradient_norm_tolerance: 1e-8,
    };

    let outcome = quench_source_escape(&surface, array![4.0, 3.0, -1.0].view(), &config);
    let SourceEscapeOutcome::Converged(record) = outcome else {
        panic!("quadratic escape did not converge: {outcome:?}")
    };

    assert_eq!(record.charged_evaluations, surface.calls());
    assert!(record.charged_evaluations > 0);
    assert!(record.minimum.gradient.dot(&record.minimum.gradient).sqrt() < 1e-8);
    let coordinate_error = &record.minimum.coordinates - &array![1.0, -2.0, 0.5];
    assert!(coordinate_error.dot(&coordinate_error) < 1e-12);
}

#[test]
fn generic_source_escape_never_exceeds_its_pes_budget() {
    let surface = CountingQuadratic::new();
    let config = SourceEscapeConfig {
        maximum_evaluations: 1,
        quench_steps: 100,
        gradient_tolerance: 1e-12,
        gradient_norm_tolerance: 1e-12,
    };

    let outcome = quench_source_escape(&surface, array![4.0, 3.0, -1.0].view(), &config);
    let SourceEscapeOutcome::Failed(failure) = outcome else {
        panic!("one evaluation unexpectedly certified a nonstationary start")
    };

    assert_eq!(failure.charged_evaluations, 1);
    assert_eq!(surface.calls(), 1);
}
