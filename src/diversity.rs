//! Annealing the distance that decides when two solutions are the same.
//!
//! Every basin-keyed mechanism in this crate compares a distance against a
//! threshold held fixed for the run. That threshold is not a constant of the
//! problem, and treating it as one is why three separate calibrations of it
//! failed to find a value that transferred.
//!
//! Lee, Lee and Scheraga make the case directly (Conformational space annealing,
//! arXiv cond-mat/0307690). Their `Dcut` "plays the role of the temperature in
//! simulated annealing": the diversity of sampling is controlled by comparing a
//! distance between two configurations against it, and "the value of `Dcut` is
//! slowly reduced just as in SA, hence the name conformational space
//! annealing". It is a schedule, not a setting, and it is the mechanism behind
//! the only published method that solves the hard Lennard-Jones sizes reliably:
//! ten independent runs finding every known global minimum up to 183 atoms,
//! against 4 runs in 1000 for a basin-hopping variant at 75 atoms.
//!
//! Two details from that paper are load-bearing and are kept here.
//!
//! The initial value is taken from the data, as half the average distance among
//! the first population, not chosen in advance. A threshold in a shape metric
//! has units of length and the right length depends on the system, which is
//! exactly what a hand-set radius cannot know.
//!
//! The reduction is slow. The threshold starts wide, so distinct-looking
//! solutions are held apart and the search stays diverse, and narrows, so
//! finer distinctions are resolved as the budget runs down. Annealing it the
//! other way, or holding it at the narrow end, collapses the population early.
//!
//! What this module supplies is the schedule. The distance measure is the
//! caller's, which for clusters is the shape distance in [`crate::shape`].

/// A distance threshold annealed from wide to narrow over a budget.
///
/// The threshold is a length in whatever metric the caller uses. It is not
/// constructed with a value: [`DiversityAnnealer::from_population`] takes it
/// from the spread of an initial population, which is what makes it a property
/// of the system rather than of this file.
#[derive(Debug, Clone)]
pub struct DiversityAnnealer {
    initial: f64,
    final_fraction: f64,
    /// Fraction of the budget over which the threshold reaches its floor.
    pub anneal_fraction: f64,
    current: f64,
    /// Times the threshold was queried.
    pub queries: usize,
}

impl DiversityAnnealer {
    /// Threshold starting at half the mean pairwise distance of a population.
    ///
    /// The factor of one half is the paper's: `Dcut` starts at `Dave / 2` where
    /// `Dave` is the average distance among the first bank.
    ///
    /// Returns `None` when fewer than two members are supplied, or when every
    /// pair is at zero distance, since neither gives a scale and a threshold
    /// invented at that point would be the hand-set constant this replaces.
    pub fn from_population<D>(members: &[usize], mut distance: D) -> Option<Self>
    where
        D: FnMut(usize, usize) -> f64,
    {
        if members.len() < 2 {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let d = distance(members[i], members[j]);
                if d.is_finite() {
                    total += d;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return None;
        }
        let mean = total / count as f64;
        if !(mean > 0.0) {
            return None;
        }
        Some(Self {
            initial: 0.5 * mean,
            final_fraction: 0.1,
            anneal_fraction: 0.8,
            current: 0.5 * mean,
            queries: 0,
        })
    }

    /// Threshold starting at a stated value, for callers with their own scale.
    pub fn from_initial(initial: f64) -> Self {
        assert!(
            initial > 0.0 && initial.is_finite(),
            "a diversity threshold must be a positive length, got {initial}"
        );
        Self {
            initial,
            final_fraction: 0.1,
            anneal_fraction: 0.8,
            current: initial,
            queries: 0,
        }
    }

    /// Sets the floor as a fraction of the initial threshold.
    pub fn with_final_fraction(mut self, fraction: f64) -> Self {
        assert!(
            fraction > 0.0 && fraction <= 1.0,
            "the floor must be a fraction of the start in (0, 1], got {fraction}"
        );
        self.final_fraction = fraction;
        self
    }

    /// Threshold at `progress` through the budget, in `[0, 1]`.
    ///
    /// Geometric rather than linear, because the threshold is a scale: halving
    /// it matters equally wherever it starts, and a linear schedule spends most
    /// of the run near the wide end and then collapses.
    pub fn threshold(&mut self, progress: f64) -> f64 {
        self.queries += 1;
        let p = (progress / self.anneal_fraction.max(1e-12)).clamp(0.0, 1.0);
        self.current = self.initial * self.final_fraction.powf(p);
        self.current
    }

    /// Most recently returned threshold, without advancing anything.
    pub fn current(&self) -> f64 {
        self.current
    }

    /// Threshold the schedule started from.
    pub fn initial(&self) -> f64 {
        self.initial
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distances of a small set laid out on a line, so the mean is known.
    fn line_distance(points: &[f64]) -> impl FnMut(usize, usize) -> f64 + '_ {
        move |i, j| (points[i] - points[j]).abs()
    }

    #[test]
    fn the_start_is_half_the_mean_pairwise_distance() {
        // Three points at 0, 1, 3: distances 1, 3, 2, mean 2, so the start is 1.
        let pts = [0.0, 1.0, 3.0];
        let a = DiversityAnnealer::from_population(&[0, 1, 2], line_distance(&pts)).unwrap();
        assert!(
            (a.initial() - 1.0).abs() < 1e-12,
            "start {} should be half the mean of 2",
            a.initial()
        );
    }

    #[test]
    fn a_population_with_no_scale_is_refused() {
        let pts = [1.0, 1.0, 1.0];
        assert!(DiversityAnnealer::from_population(&[0, 1, 2], line_distance(&pts)).is_none());
        let one = [4.0];
        assert!(DiversityAnnealer::from_population(&[0], line_distance(&one)).is_none());
    }

    #[test]
    fn the_threshold_narrows_monotonically() {
        let mut a = DiversityAnnealer::from_initial(2.0);
        let mut last = f64::INFINITY;
        for k in 0..=20 {
            let t = a.threshold(k as f64 / 20.0);
            assert!(t <= last + 1e-15, "threshold rose at {k}: {t} after {last}");
            last = t;
        }
    }

    #[test]
    fn it_starts_wide_and_reaches_its_floor() {
        let mut a = DiversityAnnealer::from_initial(2.0).with_final_fraction(0.1);
        assert!((a.threshold(0.0) - 2.0).abs() < 1e-12);
        // The floor is reached at the end of the annealing fraction, not at the
        // end of the budget, so the last stretch runs at the finest resolution.
        let at_end = a.threshold(a.anneal_fraction);
        assert!((at_end - 0.2).abs() < 1e-9, "floor {at_end} should be 0.2");
        let after = a.threshold(1.0);
        assert!((after - 0.2).abs() < 1e-9, "past the floor it should hold: {after}");
    }

    #[test]
    fn the_schedule_is_geometric_not_linear() {
        // Equal fractions of the annealing window must multiply the threshold
        // by equal factors, which is what makes it a scale rather than an
        // offset.
        let mut a = DiversityAnnealer::from_initial(1.0).with_final_fraction(0.01);
        let f = a.anneal_fraction;
        let t0 = a.threshold(0.0);
        let t1 = a.threshold(0.25 * f);
        let t2 = a.threshold(0.5 * f);
        let r1 = t1 / t0;
        let r2 = t2 / t1;
        assert!(
            (r1 - r2).abs() < 1e-9,
            "ratios {r1} and {r2} should match for a geometric schedule"
        );
    }
}
