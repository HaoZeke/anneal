//! Per-chain exploration scales learned from paid raw energy excursions.

use std::sync::Mutex;

use eindir_core::{Bounds, Objective};
use ndarray::ArrayView1;

use crate::bias::AdaptiveHeight;

pub(super) struct Temperatures {
    gaps: Vec<Option<AdaptiveHeight>>,
}

impl Temperatures {
    pub(super) fn new(replicas: usize) -> Self {
        Self {
            gaps: (0..replicas).map(|_| None).collect(),
        }
    }

    /// One scale feeds deposit tempering, acceptance and Langevin excitation.
    /// The objective's arbitrary additive constant supplies no scale evidence.
    pub(super) fn at(&self, replica: usize, generation: usize) -> f64 {
        let scale = self.gaps[replica]
            .as_ref()
            .map_or(1.0, AdaptiveHeight::gap_estimate)
            .max(1e-12);
        scale * 5.0 * std::f64::consts::LN_2 / (generation as f64 + 1.0).ln().max(1e-12)
    }

    /// One finite raw excursion supplies evidence, including rejected trials.
    /// Initial relaxation supplies its launch-to-retained energy drop. Escape
    /// learning occurs after the boundary's acceptance and coverage deposit.
    pub(super) fn observe(&mut self, replica: usize, occupied: f64, trial: f64) {
        let gap = trial - occupied;
        if !gap.is_finite() || gap <= 0.0 {
            return;
        }
        let estimate = self.gaps[replica].get_or_insert_with(|| AdaptiveHeight::new(0.5, 1.0, gap));
        estimate.observe(gap);
    }
}

/// Retain a polisher's already-paid launch value without evaluating it twice.
pub(super) struct FirstEvaluation<'a, O> {
    inner: &'a O,
    value: Mutex<Option<f64>>,
}

impl<'a, O> FirstEvaluation<'a, O> {
    pub(super) fn new(inner: &'a O) -> Self {
        Self {
            inner,
            value: Mutex::new(None),
        }
    }

    pub(super) fn energy(&self) -> Option<f64> {
        (*self.value.lock().expect("launch energy lock")).filter(|value| value.is_finite())
    }
}

impl<O: Objective<f64>> Objective<f64> for FirstEvaluation<'_, O> {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let value = self.inner.eval(x);
        let mut first = self.value.lock().expect("launch energy lock");
        if first.is_none() {
            *first = Some(value);
        }
        value
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }
}
