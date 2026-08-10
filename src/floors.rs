//! Energy floors and flicker components.
//!
//! A class is a bundle of minima that share a floor and that the hop already
//! treats as one place: union-find on accepted rises no larger than `delta_e`,
//! and assignment of a new quench to the floor whose minimum sits within
//! `delta_e` of it. That is the energy-class collapse that a global contact
//! key cannot do.
//!
//! Expected improvement of another draw from a class, under exchangeability:
//! `EI = ebar / (n + 1)`, with `ebar` the mean excess over the class minimum.
//! A class that cannot beat `E_star` even by one excess has EI zero. Saturated
//! when `P(record) = 1/(n+1)` is at most the cost-asymmetric `tau` *or* EI
//! is zero.

use crate::screen::cost_asymmetric_threshold;

/// One energy class.
#[derive(Debug, Clone)]
pub struct Floor {
    /// Lowest quenched energy in the class.
    pub e_min: f64,
    /// Second-lowest, or `e_min` when `n < 2`.
    pub e_second: f64,
    /// Members observed.
    pub n: usize,
    /// Sum of `E_i - e_min` after each update (recomputed when `e_min` falls).
    excess_sum: f64,
    /// Sum of raw energies, so a new record can rebuild the excess.
    energy_sum: f64,
}

impl Floor {
    fn new(e: f64) -> Self {
        Self {
            e_min: e,
            e_second: e,
            n: 1,
            excess_sum: 0.0,
            energy_sum: e,
        }
    }

    fn observe(&mut self, e: f64) {
        self.n += 1;
        self.energy_sum += e;
        if e < self.e_min {
            self.e_second = self.e_min;
            self.e_min = e;
            self.excess_sum = self.energy_sum - self.e_min * self.n as f64;
        } else {
            if e < self.e_second || self.n == 2 {
                self.e_second = e;
            }
            self.excess_sum += e - self.e_min;
        }
    }

    /// Mean excess over the class minimum. Zero until a second member.
    pub fn mean_excess(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.excess_sum / (self.n as f64)
        }
    }

    /// Exchangeable record probability.
    pub fn record_prob(&self) -> f64 {
        1.0 / (self.n as f64 + 1.0)
    }

    /// Non-parametric EI of another draw from this class.
    pub fn ei(&self) -> f64 {
        self.mean_excess() * self.record_prob()
    }

    /// Whether a record of typical size could undercut `e_star`.
    pub fn can_beat(&self, e_star: f64) -> bool {
        if self.e_min <= e_star {
            return true;
        }
        let room = self.mean_excess();
        room > 0.0 && self.e_min - e_star <= room
    }

    /// EI that can change the reported answer.
    pub fn useful_ei(&self, e_star: f64) -> f64 {
        if self.can_beat(e_star) {
            self.ei()
        } else {
            0.0
        }
    }

    /// Whether another full quench of this class fails the cost gate.
    pub fn saturated(&self, tau: f64, e_star: f64) -> bool {
        if !self.can_beat(e_star) {
            return true;
        }
        self.record_prob() <= tau && self.n >= 2
    }
}

/// Book of energy classes.
#[derive(Debug, Clone, Default)]
pub struct FloorBook {
    floors: Vec<Floor>,
    /// Accepted-uphill rises, for `delta_e`.
    rises: Vec<f64>,
}

impl FloorBook {
    /// Empty book.
    pub fn new() -> Self {
        Self::default()
    }

    /// Quantile of accepted uphill rises. Zero before any rise is recorded.
    pub fn delta_e(&self) -> f64 {
        if self.rises.is_empty() {
            return 0.0;
        }
        let mut v = self.rises.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Median: the rise the hop already pays. No separate knob.
        v[v.len() / 2]
    }

    /// Record an accepted uphill rise (non-positive ignored).
    pub fn observe_rise(&mut self, rise: f64) {
        if rise.is_finite() && rise > 0.0 {
            self.rises.push(rise);
        }
    }

    /// Assign a quenched energy to a class, creating one if needed.
    ///
    /// `prev` is the class the chain stood on. A rise no larger than `delta_e`
    /// stays in `prev`. Otherwise the nearest floor within `delta_e` in energy
    /// claims it.
    pub fn assign(&mut self, e: f64, prev: Option<usize>, rise: f64) -> usize {
        let de = self.delta_e();
        if let Some(p) = prev {
            if p < self.floors.len() && rise <= de.max(0.0) {
                self.floors[p].observe(e);
                return p;
            }
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, f) in self.floors.iter().enumerate() {
            let d = (e - f.e_min).abs();
            if d <= de || (de == 0.0 && d == 0.0) {
                match best {
                    Some((_, bd)) if d >= bd => {}
                    _ => best = Some((i, d)),
                }
            }
        }
        if let Some((i, _)) = best {
            self.floors[i].observe(e);
            return i;
        }
        self.floors.push(Floor::new(e));
        self.floors.len() - 1
    }

    /// Floor `i`.
    pub fn get(&self, i: usize) -> Option<&Floor> {
        self.floors.get(i)
    }

    /// Number of classes.
    pub fn len(&self) -> usize {
        self.floors.len()
    }

    /// Whether the book has no classes.
    pub fn is_empty(&self) -> bool {
        self.floors.is_empty()
    }

    /// Index of the class with largest useful EI, if any is unsaturated.
    pub fn best_start(&self, tau: f64, e_star: f64) -> Option<usize> {
        self.floors
            .iter()
            .enumerate()
            .filter(|(_, f)| !f.saturated(tau, e_star))
            .max_by(|(_, a), (_, b)| {
                a.useful_ei(e_star)
                    .partial_cmp(&b.useful_ei(e_star))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Default cost-asymmetric `tau` from the hop's step counts.
    pub fn tau(screen_steps: usize, relax_steps: usize) -> f64 {
        cost_asymmetric_threshold(screen_steps, relax_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isomers_at_one_floor_saturate() {
        let mut b = FloorBook::new();
        // No rises yet: each energy would be its own floor. Seed delta_e by
        // recording the within-floor scatter as accepted rises.
        for _ in 0..5 {
            b.observe_rise(0.02);
        }
        let id = b.assign(-396.0, None, 0.0);
        for k in 1..8 {
            let j = b.assign(-396.0 + 0.01 * k as f64, Some(id), 0.01);
            assert_eq!(j, id, "isomer opened a new floor");
        }
        let f = b.get(id).unwrap();
        assert!(f.n >= 8);
        let tau = FloorBook::tau(25, 200);
        assert!(
            f.saturated(tau, -400.0),
            "n={} p={} tau={tau} still unsaturated against a much deeper star",
            f.n,
            f.record_prob()
        );
        // Cannot beat a much deeper incumbent: saturated regardless of n.
        let mut shallow = FloorBook::new();
        shallow.observe_rise(0.01);
        let s = shallow.assign(-100.0, None, 0.0);
        shallow.assign(-99.99, Some(s), 0.01);
        assert!(shallow.get(s).unwrap().saturated(tau, -400.0));
    }

    #[test]
    fn record_updates_the_floor() {
        let mut b = FloorBook::new();
        b.observe_rise(1.0);
        let id = b.assign(-10.0, None, 0.0);
        b.assign(-12.0, Some(id), 0.5);
        let f = b.get(id).unwrap();
        assert!((f.e_min + 12.0).abs() < 1e-12);
        assert!(f.mean_excess() > 0.0);
        assert!(f.can_beat(-12.0));
    }

    #[test]
    fn ei_falls_as_one_over_n() {
        let mut f = Floor::new(0.0);
        f.observe(2.0);
        let e1 = f.ei();
        f.observe(2.0);
        f.observe(2.0);
        f.observe(2.0);
        let e2 = f.ei();
        assert!(e2 < e1, "EI did not fall: {e1} then {e2}");
    }
}
