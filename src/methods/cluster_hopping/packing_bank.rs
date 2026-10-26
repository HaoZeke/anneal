//! A small bank of minima kept apart by common-neighbour fractions.
//!
//! Population basin hopping keeps a new minimum when it is dissimilar from
//! the one it most resembles, and replaces that one when it is similar and
//! lower. The coordinate is the share of bonded pairs that are `(5,5,5)`,
//! `(4,2,1)` and `(4,2,2)`. The fourth Steinhardt parameter separates the
//! two 75-point funnels by 0.023, which is thinner than a deposition width,
//! so it is not the coordinate. The gap below is the same 0.12 that
//! [`crate::soap::ih_dominated`] already uses to call a structure
//! icosahedral: two minima at least that far apart in L1 are different
//! packings, and the test does not name a size.
//!
//! Grosso, A.; Locatelli, M.; Schoen, F. *Math. Program.* **2007**, *110*,
//! 373. The bank is a `Vec` updated in the order minima are offered, so the
//! result does not depend on a thread count.

use ndarray::{Array1, ArrayView1};

use crate::structure::cna;

/// L1 gap above which two common-neighbour vectors are different packings.
pub const GAP: f64 = 0.12;

/// Seats in one bank. Four is enough for the packings a short run actually
/// occupies, and the cap does not grow with the number of points.
const SEATS: usize = 4;

struct Seat {
    energy: f64,
    order: [f64; 3],
    state: Array1<f64>,
}

/// Minima the chain has already quenched, separated by packing.
#[derive(Default)]
pub struct PackingBank {
    seats: Vec<Seat>,
}

impl PackingBank {
    pub fn new() -> Self {
        Self { seats: Vec::new() }
    }

    /// `(555, 421, 422)` fractions at `cutoff`.
    pub fn order(state: ArrayView1<f64>, cutoff: f64) -> [f64; 3] {
        let n = state.len() / 3;
        if n < 3 || state.len() != 3 * n || !(cutoff.is_finite() && cutoff > 0.0) {
            return [0.0; 3];
        }
        let counts = cna(state, n, cutoff);
        [
            counts.fraction((5, 5, 5)),
            counts.fraction((4, 2, 1)),
            counts.fraction((4, 2, 2)),
        ]
    }

    /// Insert `state` under the population rule.
    ///
    /// A minimum within [`GAP`] of a seat replaces that seat when it is
    /// lower. A minimum farther than [`GAP`] from every seat fills an empty
    /// seat, or replaces the highest seat when the bank is full and the new
    /// minimum is lower than that seat.
    pub fn offer(&mut self, energy: f64, state: ArrayView1<f64>, cutoff: f64) {
        if !energy.is_finite() {
            return;
        }
        let n = state.len() / 3;
        if n < 3 || state.len() != 3 * n {
            return;
        }
        let order = Self::order(state, cutoff);
        if let Some(index) = self.closest(order) {
            if l1(self.seats[index].order, order) <= GAP {
                if energy < self.seats[index].energy {
                    self.seats[index] = Seat {
                        energy,
                        order,
                        state: state.to_owned(),
                    };
                }
                return;
            }
        }
        let seat = Seat {
            energy,
            order,
            state: state.to_owned(),
        };
        if self.seats.len() < SEATS {
            self.seats.push(seat);
            return;
        }
        let worst = self
            .seats
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index);
        if let Some(index) = worst
            && energy < self.seats[index].energy
        {
            self.seats[index] = seat;
        }
    }

    /// The lowest seat whose order is more than [`GAP`] from `here` and whose
    /// energy is below `energy`.
    ///
    /// The chain only returns to a packing it has left when that packing is
    /// the better one. A higher seat stays in the bank and is not installed.
    pub fn better_other(
        &self,
        here: ArrayView1<f64>,
        energy: f64,
        cutoff: f64,
    ) -> Option<(f64, Array1<f64>)> {
        let order = Self::order(here, cutoff);
        self.seats
            .iter()
            .filter(|seat| l1(seat.order, order) > GAP && seat.energy < energy - 1e-9)
            .min_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|seat| (seat.energy, seat.state.clone()))
    }

    fn closest(&self, order: [f64; 3]) -> Option<usize> {
        self.seats
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                l1(a.order, order)
                    .partial_cmp(&l1(b.order, order))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
    }
}

fn l1(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn icosahedron13() -> Array1<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        let s = 1.0 / (1.0 + p * p).sqrt();
        let mut x = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    /// A similar, lower minimum replaces its seat. A dissimilar one is kept
    /// beside it, and the chain returns only to the one that is lower.
    #[test]
    fn similar_replaces_and_dissimilar_is_kept() {
        let ico = icosahedron13();
        let cutoff = 1.2;
        let mut bank = PackingBank::new();
        bank.offer(-10.0, ico.view(), cutoff);
        bank.offer(-12.0, ico.view(), cutoff);
        assert_eq!(bank.seats.len(), 1);
        assert!((bank.seats[0].energy - -12.0).abs() < 1e-12);

        let mut other = ico.clone();
        for i in 0..13 {
            other[3 * i] += 3.0 * i as f64;
        }
        let gap = l1(
            PackingBank::order(ico.view(), cutoff),
            PackingBank::order(other.view(), cutoff),
        );
        assert!(
            gap > GAP,
            "the torn cluster should leave the icosahedral class, gap {gap}"
        );
        bank.offer(-11.0, other.view(), cutoff);
        assert_eq!(bank.seats.len(), 2);
        let jumped = bank.better_other(ico.view(), -10.0, cutoff);
        assert!(
            jumped.is_some(),
            "a lower dissimilar seat is a place to return to"
        );
        let stayed = bank.better_other(ico.view(), -12.0, cutoff);
        assert!(
            stayed.is_none(),
            "a higher seat is not installed over the incumbent"
        );
    }

    /// Relabeling points does not change the fractions, so the bank keys the
    /// packing rather than the order the coordinates were written in.
    #[test]
    fn the_order_is_invariant_under_relabeling() {
        let ico = icosahedron13();
        let mut swapped = ico.clone();
        for k in 0..3 {
            let a = swapped[k];
            swapped[k] = swapped[3 + k];
            swapped[3 + k] = a;
        }
        let a = PackingBank::order(ico.view(), 1.2);
        let b = PackingBank::order(swapped.view(), 1.2);
        assert!(l1(a, b) < 1e-12, "{a:?} against {b:?}");
    }
}
