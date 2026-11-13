//! Distinct minima of a rigid chain, and the stall destination among
//! those minima.
//!
//! The chain keeps the eight lowest distinct minima of its quenched
//! start, accepted quenches and lowest recorded minimum. Each
//! stall follows 1000 hops with no new lowest minimum. At each stall the
//! walker moves to the lowest kept minimum that is neither its own nor
//! one of the last four stall destinations. When no kept minimum
//! qualifies, the walker stays put.
//!
//! [`SEATS`], [`MEMORY`] and [`STAGNATION_HOPS`] set those three counts,
//! and [`same_minimum`] decides whether two energies are one minimum.
//! Any tie goes to the minimum offered first, so the choice reads no
//! random number and no target.
//!
//! That choice of minimum is the population step of Grosso, Locatelli
//! and Schoen (*Math. Program.* **2007**, *110*, 373), restricted to the
//! chain's own history. The same paper supplies the cluster-hopping
//! packing bank's exit from an icosahedral plateau.
//!
//! [`STAGNATION_HOPS`]: crate::methods::rigid_quench::STAGNATION_HOPS

use std::collections::VecDeque;

use ndarray::{Array1, ArrayView1};

/// Distinct minima the archive keeps.
pub const SEATS: usize = 8;

/// Distinct oxygen cages the rigid stall keeps.
///
/// Proton isomers of one cage share a seat, so this is a count of cages
/// rather than of energies. Eight seats filled with isomers of the lowest
/// cage and left the walker with no other cage to continue from.
pub const CAGE_SEATS: usize = 32;

/// Stalls for which a taken destination stays barred.
pub const MEMORY: usize = 4;

/// Whether two quenched energies are the same minimum.
///
/// Rigid quenches of one minimum agree to about one part in 1e9, so a
/// tolerance of one part in 1e8 joins those quenches. The same tolerance
/// still keeps apart minima that differ in the eighth significant figure.
pub fn same_minimum(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1.0e-8 * a.abs().max(b.abs()).max(1.0)
}

struct Seat {
    energy: f64,
    offered: usize,
    /// Oxygen-cage key, when the seat was offered as a cage.
    key: Option<u64>,
    state: Array1<f64>,
}

/// The lowest distinct minima of one chain, and the chain's recent destinations.
pub struct StallArchive {
    seats: Vec<Seat>,
    capacity: usize,
    memory: usize,
    offered: usize,
    recent: VecDeque<f64>,
}

impl Default for StallArchive {
    fn default() -> Self {
        Self::new(SEATS, MEMORY)
    }
}

impl StallArchive {
    pub fn new(capacity: usize, memory: usize) -> Self {
        Self {
            seats: Vec::new(),
            capacity: capacity.max(1),
            memory,
            offered: 0,
            recent: VecDeque::new(),
        }
    }

    /// Keep `state` when `state` is a new distinct minimum among the lowest,
    /// or a lower copy of a kept minimum.
    pub fn offer(&mut self, energy: f64, state: ArrayView1<f64>) {
        if !energy.is_finite() || state.is_empty() {
            return;
        }
        self.offered += 1;
        if let Some(seat) = self
            .seats
            .iter_mut()
            .find(|seat| same_minimum(seat.energy, energy))
        {
            if energy < seat.energy {
                seat.energy = energy;
                seat.state = state.to_owned();
            }
            return;
        }
        let seat = Seat {
            energy,
            offered: self.offered,
            key: None,
            state: state.to_owned(),
        };
        self.insert(seat);
    }

    /// Keep the lowest structure of oxygen cage `key`.
    ///
    /// Two proton arrangements of one cage share `key` and one seat. A
    /// different cage keeps its own seat even when its energy is higher,
    /// until the archive is full and that energy is the worst one held.
    pub fn offer_cage(&mut self, energy: f64, state: ArrayView1<f64>, key: u64) {
        if !energy.is_finite() || state.is_empty() {
            return;
        }
        self.offered += 1;
        if let Some(seat) = self.seats.iter_mut().find(|seat| seat.key == Some(key)) {
            if energy < seat.energy {
                seat.energy = energy;
                seat.state = state.to_owned();
            }
            return;
        }
        self.insert(Seat {
            energy,
            offered: self.offered,
            key: Some(key),
            state: state.to_owned(),
        });
    }

    /// The lowest kept cage other than `key`.
    ///
    /// The chosen destination is barred for the next [`MEMORY`] stalls.
    /// `None` means every other cage is barred, or the archive holds no
    /// other cage, so the walker stays put.
    pub fn escape_other_cage(&mut self, key: u64) -> Option<(f64, Array1<f64>)> {
        let chosen = self
            .seats
            .iter()
            .filter(|seat| seat.key != Some(key))
            .filter(|seat| !self.recent.iter().any(|&e| same_minimum(e, seat.energy)))
            .min_by(|a, b| {
                a.energy
                    .total_cmp(&b.energy)
                    .then(a.offered.cmp(&b.offered))
            })
            .map(|seat| (seat.energy, seat.state.clone()));
        if let Some((energy, _)) = chosen.as_ref()
            && self.memory > 0
        {
            self.recent.push_back(*energy);
            while self.recent.len() > self.memory {
                self.recent.pop_front();
            }
        }
        chosen
    }

    fn insert(&mut self, seat: Seat) {
        if self.seats.len() < self.capacity {
            self.seats.push(seat);
            return;
        }
        if let Some(highest) = self.highest()
            && seat.energy < self.seats[highest].energy
        {
            self.seats[highest] = seat;
        }
    }

    /// The stall destination for a walker at `here`. The next [`MEMORY`]
    /// stalls bar the chosen destination.
    ///
    /// A `None` result means every kept minimum is the walker's own or
    /// barred, so the walker stays put.
    pub fn escape(&mut self, here: f64) -> Option<(f64, Array1<f64>)> {
        let chosen = self
            .seats
            .iter()
            .filter(|seat| !same_minimum(seat.energy, here))
            .filter(|seat| !self.recent.iter().any(|&e| same_minimum(e, seat.energy)))
            .min_by(|a, b| {
                a.energy
                    .total_cmp(&b.energy)
                    .then(a.offered.cmp(&b.offered))
            })
            .map(|seat| (seat.energy, seat.state.clone()));
        if let Some((energy, _)) = chosen.as_ref()
            && self.memory > 0
        {
            self.recent.push_back(*energy);
            while self.recent.len() > self.memory {
                self.recent.pop_front();
            }
        }
        chosen
    }

    fn highest(&self) -> Option<usize> {
        self.seats
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.energy
                    .total_cmp(&b.energy)
                    .then(a.offered.cmp(&b.offered))
            })
            .map(|(index, _)| index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn archive(energies: &[f64], capacity: usize, memory: usize) -> StallArchive {
        let mut archive = StallArchive::new(capacity, memory);
        for (index, &energy) in energies.iter().enumerate() {
            archive.offer(energy, array![index as f64].view());
        }
        archive
    }

    #[test]
    fn a_stall_takes_the_lowest_minimum_that_is_not_the_walkers_own() {
        let mut kept = archive(&[-3.0, -5.0, -4.0], 8, 0);
        assert_eq!(kept.escape(-3.0).map(|(e, _)| e), Some(-5.0));
        assert_eq!(kept.escape(-5.0).map(|(e, _)| e), Some(-4.0));
    }

    #[test]
    fn a_destination_is_barred_for_the_memory_and_then_returns() {
        let mut kept = archive(&[-5.0, -4.0, -3.0, -1.0], 8, 2);
        assert_eq!(kept.escape(-1.0).map(|(e, _)| e), Some(-5.0));
        assert_eq!(kept.escape(-1.0).map(|(e, _)| e), Some(-4.0));
        assert_eq!(kept.escape(-1.0).map(|(e, _)| e), Some(-3.0));
        assert_eq!(kept.escape(-1.0).map(|(e, _)| e), Some(-5.0));
    }

    #[test]
    fn with_nothing_left_the_walker_stays() {
        let mut empty = StallArchive::default();
        assert!(empty.escape(-1.0).is_none());
        let mut only_here = archive(&[-2.0], 8, 4);
        assert!(only_here.escape(-2.0).is_none());
        let mut barred = archive(&[-2.0, -1.0], 8, 4);
        assert!(barred.escape(-1.0).is_some());
        assert!(barred.escape(-1.0).is_none());
    }

    #[test]
    fn one_minimum_quenched_twice_is_one_seat_holding_the_lower_copy() {
        let mut kept = StallArchive::new(8, 0);
        kept.offer(-430.0, array![1.0].view());
        kept.offer(-430.0 - 1.0e-7, array![2.0].view());
        assert_eq!(kept.seats.len(), 1);
        assert_eq!(kept.seats[0].state, array![2.0]);
        kept.offer(-430.001, array![3.0].view());
        assert_eq!(kept.seats.len(), 2);
    }

    #[test]
    fn a_full_archive_keeps_the_lowest_and_an_equal_copy_keeps_the_first_state() {
        let kept = archive(&[-1.0, -2.0, -3.0], 2, 0);
        let mut energies: Vec<f64> = kept.seats.iter().map(|seat| seat.energy).collect();
        energies.sort_by(f64::total_cmp);
        assert_eq!(energies, vec![-3.0, -2.0]);
        let mut tied = StallArchive::new(4, 0);
        tied.offer(-2.0, array![7.0].view());
        tied.offer(-2.0, array![8.0].view());
        assert_eq!(tied.escape(0.0).map(|(_, x)| x), Some(array![7.0]));
    }

    #[test]
    fn proton_isomers_of_one_cage_share_a_seat_and_another_cage_is_kept() {
        let mut kept = StallArchive::new(2, 0);
        kept.offer_cage(-623.0, array![1.0].view(), 7);
        kept.offer_cage(-623.4, array![2.0].view(), 7);
        kept.offer_cage(-610.0, array![3.0].view(), 9);
        assert_eq!(kept.seats.len(), 2);
        assert_eq!(kept.escape_other_cage(7).map(|(e, _)| e), Some(-610.0));
        assert_eq!(kept.escape_other_cage(9).map(|(e, _)| e), Some(-623.4));
    }
}
