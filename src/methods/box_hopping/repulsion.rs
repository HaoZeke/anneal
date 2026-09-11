//! Local separation from bounded clouds of paid peer samples.

use std::collections::VecDeque;

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::shared_bias::SAMPLE_WINDOW;

pub(super) enum Separation {
    Distant,
    Constrained,
    Moved(Array1<f64>),
}

pub(super) struct PeerSamples {
    sources: Vec<VecDeque<Array1<f64>>>,
}

impl PeerSamples {
    pub(super) fn new(replicas: usize) -> Self {
        Self {
            sources: (0..replicas).map(|_| VecDeque::new()).collect(),
        }
    }

    pub(super) fn receive(&mut self, source: usize, descriptor: Array1<f64>) {
        let samples = &mut self.sources[source];
        if samples.back().is_some_and(|sample| *sample == descriptor) {
            return;
        }
        if samples.len() == SAMPLE_WINDOW {
            samples.pop_front();
        }
        samples.push_back(descriptor);
    }

    fn nearest(&self, point: ArrayView1<f64>) -> Option<(f64, &Array1<f64>)> {
        self.sources
            .iter()
            .flatten()
            .map(|sample| {
                let squared_distance = point
                    .iter()
                    .zip(sample.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                (squared_distance.sqrt(), sample)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }

    /// A compact-support correction in normalized box coordinates. Accept a
    /// correction only when clearance from the entire received cloud improves.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn separate<R: Rng + ?Sized>(
        &self,
        point: ArrayView1<f64>,
        anchor: ArrayView1<f64>,
        widths: ArrayView1<f64>,
        upper: f64,
        radius: f64,
        weight: f64,
        rng: &mut R,
    ) -> Option<(Separation, bool)> {
        let (distance, neighbour) = self.nearest(point)?;
        let anchor_overlaps = self
            .nearest(anchor)
            .is_some_and(|(distance, _)| distance < radius);
        if distance >= radius {
            return Some((Separation::Distant, anchor_overlaps));
        }
        let mut direction = &point - neighbour;
        let mut norm = distance;
        if norm == 0.0 {
            direction = &point - &anchor;
            norm = direction.dot(&direction).sqrt();
            if norm == 0.0 {
                for (component, width) in direction.iter_mut().zip(widths.iter()) {
                    *component = if *width > 0.0 {
                        StandardNormal.sample(rng)
                    } else {
                        0.0
                    };
                }
                norm = direction.dot(&direction).sqrt();
            }
        }
        if norm == 0.0 || !norm.is_finite() {
            return Some((Separation::Constrained, anchor_overlaps));
        }
        let increment = (radius - distance) * weight.min(1.0);
        let candidate =
            Array1::from_iter(point.iter().zip(direction.iter()).zip(widths.iter()).map(
                |((coordinate, direction), width)| {
                    if *width > 0.0 {
                        (coordinate + (direction / norm) * increment).clamp(0.0, upper)
                    } else {
                        0.0
                    }
                },
            ));
        if self
            .nearest(candidate.view())
            .is_some_and(|(clearance, _)| clearance > distance)
        {
            Some((Separation::Moved(candidate), anchor_overlaps))
        } else {
            Some((Separation::Constrained, anchor_overlaps))
        }
    }
}
