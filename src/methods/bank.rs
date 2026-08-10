//! A population held apart by an annealed distance, after conformational space
//! annealing.
//!
//! Lee, Lee and Scheraga, arXiv cond-mat/0307690.
//!
//! This is the mechanism behind the only published results that solve the hard
//! Lennard-Jones sizes reliably: ten independent runs finding every known global
//! minimum up to 183 points, against 4 runs in 1000 for a basin-hopping variant
//! at 75 points. What carries it is not the perturbation operator. It is the
//! replacement rule and the schedule on `Dcut`, which "plays the role of the
//! temperature in simulated annealing".
//!
//! The rule has one idea in it. A new solution is compared against the member
//! it most resembles, not against the worst member. If the two are closer than
//! `Dcut` they are the same solution as far as the search is concerned, and only
//! the better of them is kept; if the new one resembles nothing in the bank it
//! is a genuinely different region and it displaces the worst member instead.
//! A bank under that rule cannot collapse onto one funnel, which is the failure
//! a low-temperature chain has no defence against.
//!
//! `Dcut` starts wide, so distinct-looking solutions are held apart and the
//! search stays broad, and narrows, so finer distinctions are resolved as the
//! budget runs down. The schedule is [`crate::diversity`]; the distance is the
//! caller's, and for clusters it is the shape distance in [`crate::shape`].
//!
//! # What this is not
//!
//! The published method perturbs by cutting one solution and splicing in part
//! of another. Nothing here does that: the perturbation is the caller's, and in
//! this crate it is the move library and the biased chain in
//! [`crate::methods::cluster_hopping`]. The bank supplies the diversity control
//! and nothing else, which is the part the results rest on.

use ndarray::{Array1, ArrayView1};

/// What happened to a candidate offered to the bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Better than the member it resembles, and took its place.
    Improved(usize),
    /// Resembles a member and is not better; discarded.
    Duplicate(usize),
    /// Resembles nothing in the bank, and displaced the worst member.
    Displaced(usize),
    /// Resembles nothing and the bank had room.
    Added(usize),
    /// Resembles nothing, the bank is full, and it is worse than every member.
    Rejected,
}

/// A member of the bank.
#[derive(Debug, Clone)]
pub struct Member {
    /// The solution.
    pub state: Array1<f64>,
    /// Its objective value.
    pub energy: f64,
    /// Times a candidate was found to resemble it.
    pub hits: usize,
}

/// A population under the conformational-space-annealing replacement rule.
pub struct Bank {
    members: Vec<Member>,
    /// The seeding population, kept unchanged for the whole run.
    ///
    /// Lee, Lee and Scheraga keep a copy of the first bank and draw
    /// perturbation partners from "either the first bank or the bank". It is
    /// not a detail. Without it every member is free to descend, and at 75
    /// points a bank of thirty ended holding structures between -396.28 and
    /// -396.19: thirty icosahedral variants, each distinct under the threshold
    /// and all in one funnel, with nothing left in the population to mix
    /// against. The first bank is the part of the population that cannot
    /// collapse, because nothing ever writes to it.
    first: Vec<Member>,
    capacity: usize,
    /// Current `Dcut`. Set by the caller from a [`crate::diversity`] schedule.
    pub dcut: f64,
    /// Candidates offered.
    pub offered: usize,
    /// Candidates that resembled nothing in the bank.
    pub novel: usize,
}

impl Bank {
    /// An empty bank holding at most `capacity` members.
    pub fn new(capacity: usize, dcut: f64) -> Self {
        assert!(capacity > 0, "a bank holds at least one solution");
        assert!(
            dcut > 0.0 && dcut.is_finite(),
            "Dcut is a distance and must be positive, got {dcut}"
        );
        Self {
            members: Vec::with_capacity(capacity),
            first: Vec::with_capacity(capacity),
            capacity,
            dcut,
            offered: 0,
            novel: 0,
        }
    }

    /// The members, in the order they occupy their slots.
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// How many solutions the bank holds.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the bank holds nothing.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The lowest member, if any.
    pub fn best(&self) -> Option<&Member> {
        self.members.iter().min_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Index of the highest-energy member.
    fn worst_index(&self) -> Option<usize> {
        self.members
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.energy
                    .partial_cmp(&b.1.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Offers a solution to the bank.
    ///
    /// `distance` measures how far the candidate is from a member, and is the
    /// caller's: for clusters it is a shape distance, which makes `Dcut` a
    /// length. The comparison is against the *nearest* member, which is the
    /// whole rule. Comparing against the worst instead lets a bank fill with
    /// near-copies of one good solution, and a bank in one funnel searches one
    /// funnel.
    pub fn offer<D>(&mut self, state: ArrayView1<f64>, energy: f64, mut distance: D) -> Admission
    where
        D: FnMut(ArrayView1<f64>, ArrayView1<f64>) -> f64,
    {
        self.offered += 1;
        let nearest = self
            .members
            .iter()
            .enumerate()
            .map(|(i, m)| (i, distance(state, m.state.view())))
            .filter(|(_, d)| d.is_finite())
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((i, d)) = nearest {
            if d <= self.dcut {
                self.members[i].hits += 1;
                return if energy < self.members[i].energy {
                    self.members[i].state = state.to_owned();
                    self.members[i].energy = energy;
                    Admission::Improved(i)
                } else {
                    Admission::Duplicate(i)
                };
            }
        }

        self.novel += 1;
        if self.members.len() < self.capacity {
            self.members.push(Member {
                state: state.to_owned(),
                energy,
                hits: 0,
            });
            return Admission::Added(self.members.len() - 1);
        }
        match self.worst_index() {
            Some(w) if energy < self.members[w].energy => {
                self.members[w] = Member {
                    state: state.to_owned(),
                    energy,
                    hits: 0,
                };
                Admission::Displaced(w)
            }
            _ => Admission::Rejected,
        }
    }

    /// Adds a solution without applying the replacement rule.
    ///
    /// For the seeding phase only, and it is not a convenience. `Dcut` is meant
    /// to come from the spread of the first population, so the first population
    /// cannot be filtered by a `Dcut`: with a placeholder threshold wide enough
    /// to admit anything, every seed after the first resembles the first, the
    /// bank ends the phase holding one member, and there is no spread to
    /// measure. Measured on LJ38, eight seeding chains left a bank of one.
    ///
    /// Returns `false` when the bank is full, which ends the phase.
    pub fn seed(&mut self, state: ArrayView1<f64>, energy: f64) -> bool {
        if self.members.len() >= self.capacity {
            return false;
        }
        self.offered += 1;
        self.novel += 1;
        let m = Member {
            state: state.to_owned(),
            energy,
            hits: 0,
        };
        self.first.push(m.clone());
        self.members.push(m);
        true
    }

    /// The seeding population, unchanged since the run began.
    pub fn first_bank(&self) -> &[Member] {
        &self.first
    }

    /// Picks a member to search from next.
    ///
    /// Least-used first, breaking ties by energy. The bank is a set of regions
    /// to explore, not a ranking, so spending every start on the current best
    /// defeats the point of holding the others; and a member nothing has ever
    /// been found near is the one whose surroundings are least known.
    pub fn next_start(&self) -> Option<usize> {
        self.members
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.hits.cmp(&b.1.hits).then_with(|| {
                    a.1.energy
                        .partial_cmp(&b.1.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|(i, _)| i)
    }

    /// Marks a member as having been searched from.
    pub fn mark_used(&mut self, i: usize) {
        if let Some(m) = self.members.get_mut(i) {
            m.hits += 1;
        }
    }

    /// Mean pairwise distance among the members, for setting the initial
    /// `Dcut` from the data rather than by hand.
    pub fn mean_distance<D>(&self, mut distance: D) -> Option<f64>
    where
        D: FnMut(ArrayView1<f64>, ArrayView1<f64>) -> f64,
    {
        if self.members.len() < 2 {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..self.members.len() {
            for j in (i + 1)..self.members.len() {
                let d = distance(self.members[i].state.view(), self.members[j].state.view());
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
        if mean > 0.0 { Some(mean) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn point(v: f64) -> Array1<f64> {
        Array1::from(vec![v])
    }

    fn line(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        (a[0] - b[0]).abs()
    }

    #[test]
    fn a_bank_with_room_takes_anything_far_enough_away() {
        let mut b = Bank::new(4, 1.0);
        assert_eq!(b.offer(point(0.0).view(), 5.0, line), Admission::Added(0));
        assert_eq!(b.offer(point(3.0).view(), 7.0, line), Admission::Added(1));
        assert_eq!(b.len(), 2);
    }

    /// The rule the method rests on: a candidate is judged against what it
    /// resembles, so a near-copy cannot take a slot from a distinct solution
    /// however good it is.
    #[test]
    fn a_near_copy_replaces_its_own_kind_and_no_one_else() {
        let mut b = Bank::new(2, 1.0);
        b.offer(point(0.0).view(), 5.0, line);
        b.offer(point(10.0).view(), 6.0, line);
        // Much better than everything, and half a unit from the first member.
        let a = b.offer(point(0.5).view(), -100.0, line);
        assert_eq!(a, Admission::Improved(0));
        assert_eq!(b.len(), 2);
        assert!(
            b.members()
                .iter()
                .any(|m| (m.state[0] - 10.0).abs() < 1e-12),
            "the distant member was evicted by a near-copy"
        );
    }

    #[test]
    fn a_worse_near_copy_is_discarded() {
        let mut b = Bank::new(2, 1.0);
        b.offer(point(0.0).view(), 5.0, line);
        assert_eq!(
            b.offer(point(0.5).view(), 9.0, line),
            Admission::Duplicate(0)
        );
        assert_eq!(b.len(), 1);
        assert_eq!(b.members()[0].energy, 5.0);
    }

    #[test]
    fn a_full_bank_gives_up_its_worst_to_something_new() {
        let mut b = Bank::new(2, 1.0);
        b.offer(point(0.0).view(), 5.0, line);
        b.offer(point(10.0).view(), 7.0, line);
        assert_eq!(
            b.offer(point(20.0).view(), 6.0, line),
            Admission::Displaced(1)
        );
        assert!(b.members().iter().all(|m| m.energy <= 6.0));
    }

    #[test]
    fn a_full_bank_refuses_something_new_and_worse_than_all_of_it() {
        let mut b = Bank::new(2, 1.0);
        b.offer(point(0.0).view(), 5.0, line);
        b.offer(point(10.0).view(), 7.0, line);
        assert_eq!(b.offer(point(20.0).view(), 8.0, line), Admission::Rejected);
        assert_eq!(b.len(), 2);
    }

    /// Narrowing `Dcut` has to resolve solutions the wide threshold merged,
    /// which is what makes it a schedule rather than a setting.
    #[test]
    fn narrowing_dcut_separates_what_a_wide_one_merged() {
        let mut b = Bank::new(4, 5.0);
        b.offer(point(0.0).view(), 5.0, line);
        // Two units away: one solution at Dcut = 5, two at Dcut = 1.
        assert_eq!(
            b.offer(point(2.0).view(), 6.0, line),
            Admission::Duplicate(0)
        );
        assert_eq!(b.len(), 1);
        b.dcut = 1.0;
        assert_eq!(b.offer(point(2.0).view(), 6.0, line), Admission::Added(1));
        assert_eq!(b.len(), 2);
    }

    /// A bank that always restarts from its best is a chain with extra steps.
    #[test]
    fn starts_are_spread_over_the_bank_rather_than_spent_on_the_best() {
        let mut b = Bank::new(3, 1.0);
        b.offer(point(0.0).view(), -10.0, line);
        b.offer(point(10.0).view(), -1.0, line);
        b.offer(point(20.0).view(), -2.0, line);
        let mut used = vec![0usize; 3];
        for _ in 0..9 {
            let i = b.next_start().unwrap();
            used[i] += 1;
            b.mark_used(i);
        }
        assert!(
            used.iter().all(|&c| c == 3),
            "starts went {used:?} instead of evenly over the bank"
        );
    }

    /// The seeding phase has to leave a population with a spread in it, or
    /// there is nothing to take the threshold from.
    #[test]
    fn seeding_fills_the_bank_regardless_of_distance() {
        let mut b = Bank::new(4, 1e9);
        for (v, e) in [(0.0, 3.0), (0.1, 2.0), (0.2, 1.0), (0.3, 0.0)] {
            assert!(b.seed(point(v).view(), e));
        }
        assert_eq!(b.len(), 4);
        assert!(!b.seed(point(9.0).view(), -5.0), "a full bank kept seeding");
        let m = b.mean_distance(line).unwrap();
        assert!(m > 0.0, "the seeded population has no spread: {m}");
    }

    /// The population that cannot collapse. Every member of the working bank
    /// may descend into one funnel; the first bank still holds what the run
    /// started from, so there is always something to mix against.
    #[test]
    fn the_first_bank_is_never_written_to() {
        let mut b = Bank::new(3, 1.0);
        for (v, e) in [(0.0, 5.0), (10.0, 6.0), (20.0, 7.0)] {
            b.seed(point(v).view(), e);
        }
        let before: Vec<f64> = b.first_bank().iter().map(|m| m.state[0]).collect();
        // Drive every working member somewhere else and much lower.
        for _ in 0..20 {
            b.offer(point(0.2).view(), -100.0, line);
            b.offer(point(10.2).view(), -101.0, line);
            b.offer(point(20.2).view(), -102.0, line);
        }
        let after: Vec<f64> = b.first_bank().iter().map(|m| m.state[0]).collect();
        assert_eq!(before, after, "the first bank moved");
        let energies: Vec<f64> = b.first_bank().iter().map(|m| m.energy).collect();
        assert_eq!(energies, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn the_initial_threshold_comes_from_the_population() {
        let mut b = Bank::new(4, 100.0);
        for (v, e) in [(0.0, 1.0), (1.0, 2.0), (3.0, 3.0)] {
            b.offer(point(v).view(), e, line);
        }
        // Only one member survives at Dcut = 100, so there is no scale to take.
        assert_eq!(b.len(), 1);
        assert!(b.mean_distance(line).is_none());

        let mut c = Bank::new(4, 0.1);
        for (v, e) in [(0.0, 1.0), (1.0, 2.0), (3.0, 3.0)] {
            c.offer(point(v).view(), e, line);
        }
        // Distances 1, 3, 2: mean 2.
        let m = c.mean_distance(line).unwrap();
        assert!((m - 2.0).abs() < 1e-12, "mean distance {m} should be 2");
    }
}
