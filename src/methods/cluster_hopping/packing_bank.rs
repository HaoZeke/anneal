//! Packing descriptors collected for diagnostics only.
//!
//! The search never reads this bank.  Common-neighbour labels may be retained
//! for analysis, but they do not seed, prune, accept, reject, or stop a walk.

use ndarray::{Array1, ArrayView1};

const GAP: f64 = 0.12;
const SEATS: usize = 4;

struct Seat {
    energy: f64,
    order: [f64; 3],
    #[allow(dead_code)]
    state: Array1<f64>,
}

#[derive(Default)]
pub struct PackingBank {
    seats: Vec<Seat>,
}

impl PackingBank {
    pub fn new() -> Self {
        Self { seats: Vec::new() }
    }

    fn order(state: ArrayView1<f64>, cutoff: f64) -> [f64; 3] {
        let n = state.len() / 3;
        if n < 3 || state.len() != 3 * n || !(cutoff.is_finite() && cutoff > 0.0) {
            return [0.0; 3];
        }
        let (f555, f421, f422) = crate::structure::named_triplet_fractions(state, n, cutoff);
        [f555, f421, f422]
    }

    pub fn offer(&mut self, energy: f64, state: ArrayView1<f64>, cutoff: f64) {
        if !energy.is_finite() {
            return;
        }
        let n = state.len() / 3;
        if n < 3 || state.len() != 3 * n {
            return;
        }
        let order = Self::order(state, cutoff);
        if let Some(index) = self.closest(order)
            && l1(self.seats[index].order, order) <= GAP
        {
            if energy < self.seats[index].energy {
                self.seats[index] = Seat {
                    energy,
                    order,
                    state: state.to_owned(),
                };
            }
            return;
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
