//! Nonequilibrium-umbrella bridge machinery between two known minima.
//!
//! A bridge is a string of images through descriptor space between two
//! validated minima, tiling the channel with Voronoi regions that each
//! receive their own segment budget. Confined walkers record attempted
//! exits as crossings; weights transfer on every attempt and their fixed
//! point is flux balance; the forward crossing fractions compose into a
//! committor surrogate along the string. The construction follows
//! nonequilibrium umbrella sampling in its string form and direct forward
//! flux sampling in its staging, with the descriptor playing the order
//! parameter, so the machinery is engine-agnostic: any segment runner
//! that propagates and reports descriptors drives it.
//!
//! This module is pure state and arithmetic: no wire, no engine, no
//! randomness beyond a caller-supplied draw for entry selection.

use ndarray::Array1;

/// Invalid bridge construction or update input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BridgeError {
    /// A string needs at least three images: two endpoints and a channel.
    #[error("a bridge string needs at least three images")]
    TooFewImages,
    /// Endpoint descriptors must share one dimension.
    #[error("bridge endpoints have different dimensions")]
    DimensionMismatch,
    /// Endpoints must be separated to define a chord.
    #[error("bridge endpoints coincide")]
    DegenerateChord,
    /// A referenced region does not exist.
    #[error("bridge region index out of range")]
    RegionOutOfRange,
    /// Weight transfer fraction must lie in (0, 1).
    #[error("bridge transfer fraction outside (0, 1)")]
    InvalidTransfer,
}

fn distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// A string of images through descriptor space with fixed endpoints.
#[derive(Debug, Clone)]
pub struct BridgeString {
    images: Vec<Array1<f64>>,
}

impl BridgeString {
    /// The chord between two endpoint descriptors, sampled at `k + 1`
    /// images inclusive of both endpoints.
    pub fn chord(a: &Array1<f64>, b: &Array1<f64>, k: usize) -> Result<Self, BridgeError> {
        if a.len() != b.len() {
            return Err(BridgeError::DimensionMismatch);
        }
        if k < 2 {
            return Err(BridgeError::TooFewImages);
        }
        if distance(a, b) <= 0.0 {
            return Err(BridgeError::DegenerateChord);
        }
        let images = (0..=k)
            .map(|i| {
                let t = i as f64 / k as f64;
                a * (1.0 - t) + b * t
            })
            .collect();
        Ok(Self { images })
    }

    /// Number of regions, one per image.
    pub fn regions(&self) -> usize {
        self.images.len()
    }

    /// Images in order from the A endpoint to the B endpoint.
    pub fn images(&self) -> &[Array1<f64>] {
        &self.images
    }

    /// The region owning a descriptor: the nearest image.
    pub fn assign(&self, descriptor: &Array1<f64>) -> usize {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (index, image) in self.images.iter().enumerate() {
            let d = distance(image, descriptor);
            if d < best_d {
                best_d = d;
                best = index;
            }
        }
        best
    }

    /// Distance from a descriptor to the endpoint chord segment.
    ///
    /// Membership in the bridge tube is a cheap test against this
    /// distance; points outside the tube belong to ordinary search
    /// space rather than to any region.
    pub fn chord_distance(&self, descriptor: &Array1<f64>) -> f64 {
        let a = &self.images[0];
        let b = &self.images[self.images.len() - 1];
        let ab = b - a;
        let ab2 = ab.iter().map(|v| v * v).sum::<f64>();
        let t = ((descriptor - a).iter().zip(ab.iter()).map(|(x, y)| x * y))
            .sum::<f64>()
            / ab2;
        let t = t.clamp(0.0, 1.0);
        let closest = a + &(ab * t);
        distance(&closest, descriptor)
    }

    /// Move interior images toward observed walker means, smooth, and
    /// reparametrize to equal arc length. Endpoints never move: they are
    /// the validated minima the bridge connects. Returns the largest
    /// image displacement, the caller's convergence signal.
    pub fn update(&mut self, means: &[Option<Array1<f64>>], alpha: f64) -> f64 {
        let k = self.images.len();
        let entry = self.images.clone();
        // Drift toward walker means.
        for i in 1..k - 1 {
            if let Some(mean) = means.get(i).and_then(|m| m.as_ref())
                && mean.len() == self.images[i].len()
            {
                self.images[i] = &self.images[i] * (1.0 - alpha) + mean * alpha;
            }
        }
        // Three-point smoothing of the interior.
        let snapshot = self.images.clone();
        for i in 1..k - 1 {
            let laplace = (&snapshot[i - 1] + &snapshot[i + 1]) * 0.5 - &snapshot[i];
            self.images[i] = &self.images[i] + &(laplace * 0.25);
        }
        // Reparametrize to equal image spacing. The regions are Voronoi
        // cells of the images, so the geometric property that matters is
        // equal distance between consecutive images, not equal arc length
        // along the polyline. Images stay on the smoothed polyline: their
        // arc positions iterate until consecutive chord lengths equalize,
        // which avoids the corner-cutting a repeated inscribed resample
        // would apply.
        let curve = self.images.clone();
        let mut cumulative = vec![0.0_f64; k];
        for i in 1..k {
            cumulative[i] = cumulative[i - 1] + distance(&curve[i - 1], &curve[i]);
        }
        let total = cumulative[k - 1];
        if total > 0.0 {
            let point_at = |s: f64| -> Array1<f64> {
                let s = s.clamp(0.0, total);
                let mut segment = 1;
                while segment < k - 1 && cumulative[segment] < s {
                    segment += 1;
                }
                let span = cumulative[segment] - cumulative[segment - 1];
                let t = if span > 0.0 {
                    (s - cumulative[segment - 1]) / span
                } else {
                    0.0
                };
                &curve[segment - 1] * (1.0 - t) + &curve[segment] * t
            };
            let mut arcs: Vec<f64> = (0..k)
                .map(|i| total * i as f64 / (k - 1) as f64)
                .collect();
            for _ in 0..32 {
                let points: Vec<Array1<f64>> = arcs.iter().map(|s| point_at(*s)).collect();
                let mut chord = vec![0.0_f64; k];
                for i in 1..k {
                    chord[i] = chord[i - 1] + distance(&points[i - 1], &points[i]);
                }
                let chord_total = chord[k - 1];
                if chord_total <= 0.0 {
                    break;
                }
                // Piecewise-linear inverse of arc -> cumulative chord.
                let mut shift = 0.0_f64;
                let mut next = arcs.clone();
                for (i, slot) in next.iter_mut().enumerate().take(k - 1).skip(1) {
                    let target = chord_total * i as f64 / (k - 1) as f64;
                    let mut j = 1;
                    while j < k - 1 && chord[j] < target {
                        j += 1;
                    }
                    let span = chord[j] - chord[j - 1];
                    let t = if span > 0.0 {
                        (target - chord[j - 1]) / span
                    } else {
                        0.0
                    };
                    let s = arcs[j - 1] * (1.0 - t) + arcs[j] * t;
                    shift = shift.max((s - arcs[i]).abs());
                    *slot = s;
                }
                arcs = next;
                if shift < 1e-12 * total {
                    break;
                }
            }
            for i in 1..k - 1 {
                self.images[i] = point_at(arcs[i]);
            }
        }
        (1..k - 1)
            .map(|i| distance(&entry[i], &self.images[i]))
            .fold(0.0, f64::max)
    }
}

/// Region weights under the nonequilibrium umbrella update.
#[derive(Debug, Clone)]
pub struct WeightLedger {
    weights: Vec<f64>,
    transfer: f64,
    crossings: Vec<Vec<u64>>,
    launched: Vec<u64>,
}

impl WeightLedger {
    /// Uniform weights over `regions`, transferring fraction `s` on each
    /// attempted exit.
    pub fn new(regions: usize, s: f64) -> Result<Self, BridgeError> {
        if regions < 3 {
            return Err(BridgeError::TooFewImages);
        }
        if !(s > 0.0 && s < 1.0) {
            return Err(BridgeError::InvalidTransfer);
        }
        Ok(Self {
            weights: vec![1.0 / regions as f64; regions],
            transfer: s,
            crossings: vec![vec![0; regions]; regions],
            launched: vec![0; regions],
        })
    }

    /// Record a launched segment in a region.
    pub fn launch(&mut self, region: usize) -> Result<(), BridgeError> {
        let counter = self
            .launched
            .get_mut(region)
            .ok_or(BridgeError::RegionOutOfRange)?;
        *counter += 1;
        Ok(())
    }

    /// Record an attempted exit and transfer weight toward its target.
    ///
    /// The update is Dickson and Dinner's: weight moves with attempted
    /// flux, and its fixed point balances flux in and out of every
    /// region. Total weight is conserved exactly.
    pub fn crossing(&mut self, from: usize, to: usize) -> Result<(), BridgeError> {
        if from >= self.weights.len() || to >= self.weights.len() {
            return Err(BridgeError::RegionOutOfRange);
        }
        self.crossings[from][to] += 1;
        let moved = self.transfer * self.weights[from];
        self.weights[from] -= moved;
        self.weights[to] += moved;
        Ok(())
    }

    /// Current region weights, summing to one.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Forward crossing fraction out of each region toward its successor.
    pub fn forward_fractions(&self) -> Vec<f64> {
        (0..self.weights.len() - 1)
            .map(|k| {
                let launched = self.launched[k].max(1) as f64;
                self.crossings[k][k + 1] as f64 / launched
            })
            .collect()
    }

    /// Committor surrogate along the string: the probability that a
    /// segment from the A side reaches image `k` before returning,
    /// as the running product of forward fractions.
    pub fn committor_surrogate(&self) -> Vec<f64> {
        let mut q = vec![1.0];
        for p in self.forward_fractions() {
            q.push(q.last().copied().unwrap_or(1.0) * p);
        }
        q
    }

    /// How far the weights sit from flux balance: the largest absolute
    /// mismatch between weighted flux in and out of a region, with the
    /// observed crossing counts normalized to rates so the residual
    /// measures the weights rather than the length of the observation.
    /// Zero at the fixed point.
    pub fn flux_balance_residual(&self) -> f64 {
        let n = self.weights.len();
        let total: u64 = self.crossings.iter().flatten().sum();
        if total == 0 {
            return 0.0;
        }
        let mut worst = 0.0_f64;
        for k in 0..n {
            let mut inflow = 0.0;
            let mut outflow = 0.0;
            for j in 0..n {
                if j != k {
                    inflow += self.weights[j] * self.crossings[j][k] as f64 / total as f64;
                    outflow += self.weights[k] * self.crossings[k][j] as f64 / total as f64;
                }
            }
            worst = worst.max((inflow - outflow).abs());
        }
        worst
    }
}

/// Stored entry configurations for each region.
#[derive(Debug, Clone, Default)]
pub struct EntryLists {
    entries: Vec<Vec<Array1<f64>>>,
    capacity: usize,
}

impl EntryLists {
    /// Lists for `regions`, each holding at most `capacity` entries.
    pub fn new(regions: usize, capacity: usize) -> Self {
        Self {
            entries: vec![Vec::new(); regions],
            capacity: capacity.max(1),
        }
    }

    /// Record an entry configuration for a region, evicting the oldest
    /// beyond capacity.
    pub fn push(&mut self, region: usize, state: Array1<f64>) -> Result<(), BridgeError> {
        let list = self
            .entries
            .get_mut(region)
            .ok_or(BridgeError::RegionOutOfRange)?;
        list.push(state);
        if list.len() > self.capacity {
            list.remove(0);
        }
        Ok(())
    }

    /// Draw an entry for a region by a caller-supplied index draw.
    pub fn draw(&self, region: usize, draw: u64) -> Option<&Array1<f64>> {
        let list = self.entries.get(region)?;
        if list.is_empty() {
            return None;
        }
        list.get((draw % list.len() as u64) as usize)
    }

    /// Number of stored entries for a region.
    pub fn len(&self, region: usize) -> usize {
        self.entries.get(region).map_or(0, Vec::len)
    }

    /// Whether a region has no stored entry.
    pub fn is_empty(&self, region: usize) -> bool {
        self.len(region) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn endpoints() -> (Array1<f64>, Array1<f64>) {
        (array![0.0, 0.0], array![12.0, 0.0])
    }

    #[test]
    fn a_chord_string_partitions_monotonically_along_the_chord() {
        let (a, b) = endpoints();
        let string = BridgeString::chord(&a, &b, 6).unwrap();
        let mut previous = 0;
        for step in 0..=60 {
            let x = array![step as f64 * 0.2, 0.3];
            let region = string.assign(&x);
            assert!(region >= previous, "assignment regressed along the chord");
            previous = region;
        }
        assert_eq!(string.assign(&a), 0);
        assert_eq!(string.assign(&b), string.regions() - 1);
    }

    #[test]
    fn degenerate_and_mismatched_endpoints_are_rejected() {
        let a = array![1.0, 2.0];
        assert_eq!(
            BridgeString::chord(&a, &a, 6).unwrap_err(),
            BridgeError::DegenerateChord
        );
        assert_eq!(
            BridgeString::chord(&a, &array![1.0], 6).unwrap_err(),
            BridgeError::DimensionMismatch
        );
        assert_eq!(
            BridgeString::chord(&a, &array![3.0, 4.0], 1).unwrap_err(),
            BridgeError::TooFewImages
        );
    }

    #[test]
    fn chord_distance_measures_the_tube() {
        let (a, b) = endpoints();
        let string = BridgeString::chord(&a, &b, 6).unwrap();
        assert!(string.chord_distance(&array![6.0, 0.0]) < 1e-12);
        assert!((string.chord_distance(&array![6.0, 2.5]) - 2.5).abs() < 1e-12);
        // Beyond the endpoint the distance is to the endpoint, not the line.
        assert!((string.chord_distance(&array![15.0, 0.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn update_pins_endpoints_and_reparametrizes_to_equal_arcs() {
        let (a, b) = endpoints();
        let mut string = BridgeString::chord(&a, &b, 6).unwrap();
        let means: Vec<Option<Array1<f64>>> = (0..string.regions())
            .map(|i| {
                (i > 0 && i + 1 < 7).then(|| array![2.0 * i as f64, if i == 3 { 4.0 } else { 0.0 }])
            })
            .collect();
        for _ in 0..8 {
            string.update(&means, 0.3);
        }
        assert_eq!(string.images()[0], a);
        assert_eq!(string.images()[6], b);
        let spans: Vec<f64> = string
            .images()
            .windows(2)
            .map(|w| distance(&w[0], &w[1]))
            .collect();
        let mean_span = spans.iter().sum::<f64>() / spans.len() as f64;
        for span in &spans {
            assert!(
                (span - mean_span).abs() < 1e-6 * mean_span,
                "arc lengths uneven after reparametrization: {spans:?}"
            );
        }
        // The middle image drifted toward the displaced mean.
        assert!(string.images()[3][1] > 0.1);
    }

    #[test]
    fn a_converged_string_reports_vanishing_movement() {
        let (a, b) = endpoints();
        let mut string = BridgeString::chord(&a, &b, 6).unwrap();
        let means: Vec<Option<Array1<f64>>> =
            string.images().iter().cloned().map(Some).collect();
        let moved = string.update(&means, 0.5);
        assert!(moved < 1e-12, "chord at its own means still moved {moved}");
    }

    #[test]
    fn weight_is_conserved_under_arbitrary_transfers() {
        let mut ledger = WeightLedger::new(5, 0.1).unwrap();
        let hops = [(0, 1), (1, 2), (4, 3), (2, 1), (3, 4), (1, 0), (2, 3)];
        for &(from, to) in hops.iter().cycle().take(700) {
            ledger.crossing(from, to).unwrap();
        }
        let total: f64 = ledger.weights().iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "weight leaked: total {total}");
        assert!(ledger.weights().iter().all(|w| *w >= 0.0));
    }

    #[test]
    fn weights_converge_to_flux_balance_on_a_synthetic_chain() {
        // Three regions in a chain with forward flux twice the backward
        // flux: at flux balance W_{k+1}/W_k = 2, so W = (1, 2, 4)/7. The
        // sequential update carries an O(s) bias off the continuous fixed
        // point, so the transfer fraction stays small.
        let mut ledger = WeightLedger::new(3, 0.005).unwrap();
        let pattern = [(0, 1), (0, 1), (1, 0), (1, 2), (1, 2), (2, 1)];
        let mut residuals = Vec::new();
        for round in 0..2000 {
            for &(from, to) in &pattern {
                ledger.crossing(from, to).unwrap();
            }
            if round % 500 == 499 {
                residuals.push(ledger.flux_balance_residual());
            }
        }
        let w = ledger.weights();
        assert!((w[1] / w[0] - 2.0).abs() < 0.05, "W1/W0 = {}", w[1] / w[0]);
        assert!((w[2] / w[1] - 2.0).abs() < 0.05, "W2/W1 = {}", w[2] / w[1]);
        assert!(
            residuals.windows(2).all(|r| r[1] <= r[0] * 1.05),
            "flux-balance residual not decreasing: {residuals:?}"
        );
    }

    #[test]
    fn the_committor_surrogate_is_the_running_product_of_forward_fractions() {
        let mut ledger = WeightLedger::new(3, 0.1).unwrap();
        for _ in 0..4 {
            ledger.launch(0).unwrap();
            ledger.launch(1).unwrap();
        }
        ledger.crossing(0, 1).unwrap();
        ledger.crossing(0, 1).unwrap();
        ledger.crossing(1, 2).unwrap();
        let q = ledger.committor_surrogate();
        assert_eq!(q.len(), 3);
        assert!((q[0] - 1.0).abs() < 1e-12);
        assert!((q[1] - 0.5).abs() < 1e-12);
        assert!((q[2] - 0.125).abs() < 1e-12);
    }

    #[test]
    fn entry_lists_cap_and_draw_by_modulus() {
        let mut lists = EntryLists::new(3, 2);
        assert!(lists.is_empty(1));
        assert!(lists.draw(1, 7).is_none());
        for value in 0..4 {
            lists.push(1, array![value as f64]).unwrap();
        }
        assert_eq!(lists.len(1), 2);
        // Oldest evicted: survivors are 2.0 and 3.0.
        assert_eq!(lists.draw(1, 0).unwrap()[0], 2.0);
        assert_eq!(lists.draw(1, 5).unwrap()[0], 3.0);
        assert_eq!(lists.push(9, array![0.0]).unwrap_err(), BridgeError::RegionOutOfRange);
    }
}
