//! A plateau leave that any potential can turn on.
//!
//! The energy archive keeps the lowest few minima. Once a chain sits in
//! one funnel those seats fill with copies of it, and a leave from the
//! archive stays inside the funnel. This population keeps structures
//! that are far apart in their sorted pair distances, and the kick and
//! the temperature move with the acceptance of the last block.
//!
//! Off, the search never reads it. The acceptance block is the same
//! length the temperature dial already uses.

use ndarray::{Array1, ArrayView1};

use super::stall_archive::same_minimum;

/// Seats in the population. The same count the energy archive keeps.
const SEATS: usize = super::stall_archive::SEATS;

/// Hops in one acceptance block.
const BLOCK: usize = 40;

/// Step and temperature stay inside these factors of the preset.
const STEP_LO: f64 = 0.25;
const STEP_HI: f64 = 4.0;
const HEAT_LO: f64 = 1.0;
const HEAT_HI: f64 = 4.0;

/// Relative root-mean-square of the pair distances below which two
/// quenches are one structure.
const SAME_STRUCTURE: f64 = 0.02;

struct Seat {
    energy: f64,
    state: Array1<f64>,
    print: Vec<f64>,
}

/// Acceptance dial and the diverse population for one chain.
pub struct PlateauSearch {
    step: f64,
    heat: f64,
    window: usize,
    yes: usize,
    improved: usize,
    seats: Vec<Seat>,
    hold_until: usize,
    leaves: u32,
}

impl Default for PlateauSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl PlateauSearch {
    pub fn new() -> Self {
        Self {
            step: 1.0,
            heat: 1.0,
            window: 0,
            yes: 0,
            improved: 0,
            seats: Vec::new(),
            hold_until: 0,
            leaves: 0,
        }
    }

    pub fn step(&self) -> f64 {
        self.step
    }

    pub fn heat(&self) -> f64 {
        self.heat
    }

    pub fn leaves(&self) -> u32 {
        self.leaves
    }

    /// The incumbent-gap pull stays off until `spent` reaches this count.
    pub fn hold_until(&mut self, charged: usize) {
        self.hold_until = self.hold_until.max(charged);
    }

    pub fn holding(&self, spent: usize) -> bool {
        spent < self.hold_until
    }

    /// One hop. A block whose acceptance is outside a quarter of one half
    /// moves the step. A block that accepts and never lowers the incumbent
    /// raises the temperature; a block that does lower it returns the
    /// temperature toward the preset.
    pub fn observe(&mut self, accepted: bool, improved: bool) {
        self.window += 1;
        if accepted {
            self.yes += 1;
        }
        if improved {
            self.improved += 1;
        }
        if self.window < BLOCK {
            return;
        }
        let rate = self.yes as f64 / self.window as f64;
        let gains = self.improved;
        self.window = 0;
        self.yes = 0;
        self.improved = 0;
        if rate > 0.75 {
            self.step = (self.step * 1.25).min(STEP_HI);
        } else if rate < 0.25 {
            self.step = (self.step / 1.25).max(STEP_LO);
        }
        // The kick may grow when acceptance is high. It is capped, and a
        // leave may not step above the temperature window: that window is
        // what stopped LJ38 jumping from the fcc funnel into a melt. The
        // temperature rises when a block accepts and does not improve.
        if gains == 0 && rate > 0.5 {
            self.heat = (self.heat * 1.25).min(HEAT_HI);
        } else if gains > 0 {
            self.heat = (self.heat / 1.25).max(HEAT_LO);
        }
    }

    /// Keep `state` when it is a new shape, or a lower copy of one already kept.
    pub fn offer(&mut self, energy: f64, state: ArrayView1<f64>) {
        if !energy.is_finite() || state.len() < 6 {
            return;
        }
        let print = sorted_pairs(state);
        if print.is_empty() {
            return;
        }
        if let Some(seat) = self
            .seats
            .iter_mut()
            .find(|seat| same_print(&seat.print, &print))
        {
            if energy < seat.energy {
                seat.energy = energy;
                seat.state = state.to_owned();
                seat.print = print;
            }
            return;
        }
        let seat = Seat {
            energy,
            state: state.to_owned(),
            print,
        };
        if self.seats.len() < SEATS {
            self.seats.push(seat);
            return;
        }
        let nn = nearest(&seat.print, &self.seats, None);
        let lowest = lowest_index(&self.seats);
        let Some(victim) = most_crowded(&self.seats, lowest) else {
            return;
        };
        if nn > nearest(&self.seats[victim].print, &self.seats, Some(victim)) {
            self.seats[victim] = seat;
        }
    }

    /// The kept structure farthest from `state` that is not the same minimum
    /// and is not above `ceiling`.
    ///
    /// The ceiling is the incumbent plus the temperature the chain is hopping
    /// at. Without it the farthest fingerprint is an early hot structure, and
    /// the leave steps out of a finished funnel into that structure.
    pub fn leave(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        ceiling: f64,
    ) -> Option<(f64, Array1<f64>)> {
        let here = sorted_pairs(state);
        let mut chosen: Option<(f64, usize)> = None;
        for (index, seat) in self.seats.iter().enumerate() {
            if seat.energy > ceiling {
                continue;
            }
            if same_minimum(seat.energy, energy) || same_print(&seat.print, &here) {
                continue;
            }
            let distance = pair_distance(&seat.print, &here);
            if chosen.as_ref().is_none_or(|(held, _)| distance > *held) {
                chosen = Some((distance, index));
            }
        }
        let (_, index) = chosen?;
        self.leaves += 1;
        Some((self.seats[index].energy, self.seats[index].state.clone()))
    }
}

fn sorted_pairs(state: ArrayView1<f64>) -> Vec<f64> {
    let n = state.len() / 3;
    let mut distances = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut squared = 0.0;
            for k in 0..3 {
                let delta = state[3 * i + k] - state[3 * j + k];
                squared += delta * delta;
            }
            distances.push(squared.sqrt());
        }
    }
    distances.sort_by(f64::total_cmp);
    distances
}

fn pair_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn same_print(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let scale = a.iter().sum::<f64>() / a.len() as f64;
    if !(scale > 0.0) {
        return false;
    }
    pair_distance(a, b) / scale / (a.len() as f64).sqrt() < SAME_STRUCTURE
}

fn nearest(print: &[f64], seats: &[Seat], skip: Option<usize>) -> f64 {
    seats
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != skip)
        .map(|(_, seat)| pair_distance(print, &seat.print))
        .fold(f64::INFINITY, f64::min)
}

fn lowest_index(seats: &[Seat]) -> Option<usize> {
    seats
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.energy.total_cmp(&b.energy))
        .map(|(index, _)| index)
}

fn most_crowded(seats: &[Seat], protect: Option<usize>) -> Option<usize> {
    seats
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != protect)
        .min_by(|(i, _), (j, _)| {
            nearest(&seats[*i].print, seats, Some(*i))
                .total_cmp(&nearest(&seats[*j].print, seats, Some(*j)))
                .then(i.cmp(j))
        })
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn line(x: f64) -> Array1<f64> {
        array![0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, x, 0.0]
    }

    #[test]
    fn a_high_acceptance_without_improvement_grows_the_step_and_the_temperature() {
        let mut search = PlateauSearch::new();
        for _ in 0..BLOCK {
            search.observe(true, false);
        }
        assert!((search.step() - 1.25).abs() < 1e-12);
        assert!((search.heat() - 1.25).abs() < 1e-12);
        for _ in 0..8 * BLOCK {
            search.observe(true, false);
        }
        assert!(
            (search.step() - STEP_HI).abs() < 1e-12,
            "the kick grew past one and a half, {}",
            search.step()
        );
    }

    #[test]
    fn an_improvement_returns_the_temperature_toward_the_preset() {
        let mut search = PlateauSearch::new();
        for _ in 0..BLOCK {
            search.observe(true, false);
        }
        for _ in 0..BLOCK {
            search.observe(true, true);
        }
        assert!((search.heat() - 1.0).abs() < 1e-12);
        assert!(search.step() > 1.0);
    }

    #[test]
    fn a_near_copy_does_not_evict_a_distant_structure() {
        let mut search = PlateauSearch::new();
        search.offer(-1.0, line(1.0).view());
        for k in 0..SEATS {
            search.offer(-2.0 - 0.001 * k as f64, line(5.0 + 0.01 * k as f64).view());
        }
        let destination = search
            .leave(-2.004, line(5.04).view(), f64::INFINITY)
            .expect("a leave");
        assert!(
            (destination.0 + 1.0).abs() < 1e-12,
            "the leave stayed in the crowded well, energy {}",
            destination.0
        );
    }

    #[test]
    fn a_leave_does_not_step_above_the_temperature_window() {
        let mut search = PlateauSearch::new();
        search.offer(-10.0, line(3.0).view());
        search.offer(-1.0, line(12.0).view());
        assert!(
            search.leave(-10.0, line(3.0).view(), -10.0 + 0.8).is_none(),
            "the hot seat was inside the window"
        );
        let destination = search
            .leave(-10.0, line(3.0).view(), f64::INFINITY)
            .expect("the hot seat is eligible once the window is open");
        assert!((destination.0 + 1.0).abs() < 1e-12);
    }
}
