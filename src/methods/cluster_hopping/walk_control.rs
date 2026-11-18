// Copyright (c) anneal developers
//! All-atom step, and the temperature only when that step is against a
//! bound, taken from the run's own acceptance. A quiet incumbent starts
//! a basin hop that keeps its chain until it lowers the record, and stops
//! after one such quiet stretch. A chain that is still on the incumbent
//! after the burst is dropped, so the next opening draws again. A chain
//! that has already left keeps its place. The opening
//! descent, the first few incumbents, is not a plateau. A stall whose four
//! histories would start past the run is left to the library: pulling the
//! hop forward lands on searches that still succeed. A burst that does not
//! lower the record is not a quiet stretch of the library. Nothing here is
//! a size, a range, or a named packing.
//!
//! Off, the search does not consult this and draws the same numbers.

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const WINDOW: usize = 8;
const ACCEPT_MEMORY: usize = 32;
const POPULATION: usize = 16;
/// Improvements before a quiet stretch is a plateau. The opening descent
/// is a run of new incumbents, not a stall.
const CLIMB: usize = 4;
const GROW: f64 = 1.1;
const STEP_LO: f64 = 0.05;
const STEP_HI: f64 = 2.0;
const TEMP_LO: f64 = 0.25;
const TEMP_HI: f64 = 4.0;

/// Where a plateau hop starts. The incumbent, a fresh draw, or a member
/// the chain has visited least.
pub(super) enum Opening {
    Incumbent,
    Random,
    Stored { energy: f64, coords: Array1<f64> },
}

struct Member {
    fingerprint: u64,
    energy: f64,
    coords: Array1<f64>,
    visits: usize,
}

/// Acceptance memory and the population a plateau restarts from.
pub(super) struct WalkControl {
    step: f64,
    length: f64,
    temperature: f64,
    temp_scale: f64,
    window_n: usize,
    window_yes: usize,
    seen: bool,
    incumbents: usize,
    best: f64,
    spent_at_best: usize,
    last_burst_at: usize,
    failed_bursts: usize,
    burst_calls: usize,
    /// All-atom chain carried from the previous burst, and its generator.
    /// A burst that does not lower the incumbent continues this chain
    /// instead of starting again at the incumbent.
    side: Option<StdRng>,
    carry_energy: f64,
    carry: Option<Array1<f64>>,
    accepts: Vec<u64>,
    members: Vec<Member>,
    avoid: Vec<u64>,
}

impl WalkControl {
    /// Step `step` at length `length`, temperature held at `temperature`.
    pub(super) fn new(step: f64, length: f64, temperature: f64) -> Self {
        Self {
            step,
            length,
            temperature,
            temp_scale: 1.0,
            window_n: 0,
            window_yes: 0,
            seen: false,
            incumbents: 0,
            best: f64::INFINITY,
            spent_at_best: 0,
            last_burst_at: 0,
            failed_bursts: 0,
            burst_calls: 0,
            side: None,
            carry_energy: f64::INFINITY,
            carry: None,
            accepts: Vec::new(),
            members: Vec::new(),
            avoid: Vec::new(),
        }
    }

    /// Half-width of the all-atom kick.
    pub(super) fn step(&self) -> f64 {
        self.step
    }

    /// Acceptance temperature. The preset until the step is against a bound.
    pub(super) fn temperature(&self) -> f64 {
        self.temperature * self.temp_scale
    }

    /// The incumbent the ledger is holding. A lower value resets the plateau.
    pub(super) fn observe_best(&mut self, spent: usize, best: f64) {
        if !best.is_finite() {
            return;
        }
        if !self.seen {
            self.seen = true;
            self.incumbents = 1;
            self.best = best;
            self.spent_at_best = spent;
            return;
        }
        if best < self.best - 1e-12 {
            self.incumbents = self.incumbents.saturating_add(1);
            self.best = best;
            self.spent_at_best = spent;
            self.failed_bursts = 0;
            self.burst_calls = 0;
            self.accepts.clear();
            self.avoid.clear();
            self.drop_chain();
        }
    }

    /// One accepted quench of the library walk, by pair-distance fingerprint.
    pub(super) fn observe_accept(&mut self, fingerprint: u64) {
        if self.accepts.len() == ACCEPT_MEMORY {
            self.accepts.remove(0);
        }
        self.accepts.push(fingerprint);
    }

    /// The record has been quiet for four times as long as it took to reach,
    /// the climb has more than a few incumbents, the last burst is not still
    /// inside its cooldown, and the call cap for this incumbent has room.
    pub(super) fn due(&self, spent: usize) -> bool {
        if !self.seen || self.incumbents < CLIMB {
            return false;
        }
        let history = self.spent_at_best.max(1);
        let quiet = spent.saturating_sub(self.spent_at_best);
        if quiet <= history.saturating_mul(4) {
            return false;
        }
        if self.last_burst_at > 0 && spent.saturating_sub(self.last_burst_at) <= history {
            return false;
        }
        self.burst_calls < history.saturating_mul(2)
    }

    /// Calls this burst may still spend. Two histories, then the library
    /// keeps the rest of the budget.
    pub(super) fn calls_left(&self) -> usize {
        let history = self.spent_at_best.max(1);
        history.saturating_mul(2).saturating_sub(self.burst_calls)
    }

    /// Seed for the burst's own generator. The library generator is not read.
    pub(super) fn burst_seed(&self, best_bits: u64) -> u64 {
        best_bits ^ (self.failed_bursts as u64).wrapping_mul(0x9E37_79B9) ^ 0x57A1_C0A1
    }

    /// Clears the acceptance window. The step, the population and the
    /// carried chain stay.
    pub(super) fn begin_burst(&mut self) {
        self.window_n = 0;
        self.window_yes = 0;
    }

    /// Generator for this burst. The first burst is seeded; later bursts
    /// keep the generator they left.
    pub(super) fn take_rng(&mut self, best_bits: u64) -> StdRng {
        self.side
            .take()
            .unwrap_or_else(|| StdRng::seed_from_u64(self.burst_seed(best_bits)))
    }

    /// Keeps the generator for the next burst.
    pub(super) fn keep_rng(&mut self, rng: StdRng) {
        self.side = Some(rng);
    }

    /// Chain the previous burst ended on, if it did not lower the incumbent.
    pub(super) fn take_carry(&mut self) -> Option<(f64, Array1<f64>)> {
        self.carry.take().map(|coords| (self.carry_energy, coords))
    }

    /// Remembers where the all-atom chain stopped.
    pub(super) fn keep_carry(&mut self, energy: f64, coords: Array1<f64>) {
        self.carry_energy = energy;
        self.carry = Some(coords);
    }

    /// Drops the carried chain. The next burst opens again.
    pub(super) fn drop_chain(&mut self) {
        self.side = None;
        self.carry = None;
        self.carry_energy = f64::INFINITY;
    }

    /// The burst did not lower the incumbent.
    pub(super) fn note_failed(&mut self, spent: usize, calls: usize) {
        self.last_burst_at = spent;
        self.failed_bursts = self.failed_bursts.saturating_add(1);
        self.burst_calls = self.burst_calls.saturating_add(calls);
    }

    /// The burst lowered the incumbent. The next quiet stretch starts over.
    pub(super) fn note_success(&mut self, spent: usize) {
        self.last_burst_at = spent;
        self.failed_bursts = 0;
        self.burst_calls = 0;
        self.accepts.clear();
        self.drop_chain();
    }

    /// First look leaves from the incumbent. The next leaves from a fresh
    /// draw. After that, from the least-visited stored minimum when one exists.
    pub(super) fn opening(&mut self, incumbent: u64) -> Opening {
        if self.failed_bursts >= 2
            && let Some(index) = self.least_visited(incumbent)
        {
            self.members[index].visits = self.members[index].visits.saturating_add(1);
            return Opening::Stored {
                energy: self.members[index].energy,
                coords: self.members[index].coords.clone(),
            };
        }
        if self.failed_bursts.is_multiple_of(2) {
            Opening::Incumbent
        } else {
            Opening::Random
        }
    }

    /// A quenched minimum the population does not yet hold.
    pub(super) fn remember(&mut self, fingerprint: u64, energy: f64, coords: ArrayView1<f64>) {
        if let Some(member) = self
            .members
            .iter_mut()
            .find(|member| member.fingerprint == fingerprint)
        {
            member.visits = member.visits.saturating_add(1);
            if energy < member.energy {
                member.energy = energy;
                member.coords = coords.to_owned();
            }
            return;
        }
        self.members.push(Member {
            fingerprint,
            energy,
            coords: coords.to_owned(),
            visits: 1,
        });
        self.shrink();
    }

    /// Recently left, and not a new incumbent.
    pub(super) fn forbids(&self, fingerprint: u64) -> bool {
        self.avoid.contains(&fingerprint)
    }

    /// Puts a minimum just left on the avoid list.
    pub(super) fn note_avoid(&mut self, fingerprint: u64) {
        if self.avoid.len() == 20 {
            self.avoid.remove(0);
        }
        if !self.avoid.contains(&fingerprint) {
            self.avoid.push(fingerprint);
        }
    }

    /// One all-atom quench. A closed window moves the step toward one half
    /// acceptance, and the temperature only when the step cannot move.
    pub(super) fn observe_trial(&mut self, accepted: bool) {
        self.window_n += 1;
        if accepted {
            self.window_yes += 1;
        }
        if self.window_n < WINDOW {
            return;
        }
        let rate = self.window_yes as f64 / self.window_n as f64;
        self.window_n = 0;
        self.window_yes = 0;
        if (rate - 0.5).abs() <= 0.15 {
            return;
        }
        let span = self.length.max(1e-12);
        let lo = STEP_LO * span;
        let hi = STEP_HI * span;
        if rate > 0.5 {
            if self.step >= hi * (1.0 - 1e-9) {
                self.temp_scale = (self.temp_scale / GROW).max(TEMP_LO);
            } else {
                self.step = (self.step * GROW).min(hi);
            }
        } else if self.step <= lo * (1.0 + 1e-9) {
            self.temp_scale = (self.temp_scale * GROW).min(TEMP_HI);
        } else {
            self.step = (self.step / GROW).max(lo);
        }
    }

    fn least_visited(&self, incumbent: u64) -> Option<usize> {
        self.members
            .iter()
            .enumerate()
            .filter(|(_, member)| member.fingerprint != incumbent)
            .min_by_key(|(_, member)| (member.visits, member.energy.to_bits()))
            .map(|(index, _)| index)
    }

    fn shrink(&mut self) {
        while self.members.len() > POPULATION {
            let best = self
                .members
                .iter()
                .map(|member| member.energy)
                .fold(f64::INFINITY, f64::min);
            let drop = self
                .members
                .iter()
                .enumerate()
                .filter(|(_, member)| member.energy > best + 1e-12)
                .max_by_key(|(_, member)| member.visits)
                .map(|(index, _)| index);
            match drop {
                Some(index) => {
                    self.members.swap_remove(index);
                }
                None => break,
            }
        }
    }
}

/// Pair-distance histogram. The same geometry hashes the same way, and the
/// basin index is not consulted, so filing a structure does not scan the
/// minima the run has already opened.
pub(super) fn fingerprint(coords: ArrayView1<f64>, length: f64) -> u64 {
    let n = coords.len() / 3;
    if n == 0 {
        return 0;
    }
    let width = (0.05 * length).max(1e-9);
    let mut bins = [0u64; 64];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coords[3 * i + axis] - coords[3 * j + axis];
                square += delta * delta;
            }
            let bin = (square.sqrt() / width) as usize;
            if bin < bins.len() {
                bins[bin] += 1;
            }
        }
    }
    let mut hash = 0xcbf29ce484222325u64;
    for count in bins {
        hash ^= count;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn the_opening_descent_is_not_a_plateau() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        control.observe_best(1_000, -1.0);
        control.observe_best(2_000, -2.0);
        control.observe_best(3_000, -3.0);
        assert!(!control.due(100_000));
    }

    #[test]
    fn a_short_quiet_is_not_a_plateau_and_a_long_one_is() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        assert!(!control.due(0));
        control.observe_best(1, -1.0);
        control.observe_best(2, -2.0);
        control.observe_best(3, -3.0);
        control.observe_best(1_000, -10.0);
        assert!(!control.due(5_000), "quiet of four histories is the bar");
        assert!(control.due(5_001));
        control.note_failed(5_200, 200);
        assert!(!control.due(6_000), "the cooldown is the history itself");
        assert!(control.due(5_200 + 1_000 + 1));
    }

    #[test]
    fn the_quiet_stretch_is_the_call_cap_and_a_lower_best_clears_it() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        control.observe_best(1, -1.0);
        control.observe_best(2, -2.0);
        control.observe_best(3, -3.0);
        control.observe_best(100, -4.0);
        // Two histories is the cap: 200.
        control.note_failed(600, 150);
        assert!(!control.due(700), "cooldown is the history");
        assert!(control.due(701), "two histories are still open");
        control.note_failed(701, 100);
        assert!(!control.due(802), "two histories have been spent");
        control.observe_best(900, -5.0);
        assert_eq!(control.burst_calls, 0);
        assert!(!control.due(1_000));
    }

    #[test]
    fn a_quiet_stretch_inside_four_histories_is_not_pulled_forward() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        control.observe_best(1, -1.0);
        control.observe_best(2, -2.0);
        control.observe_best(3, -3.0);
        control.observe_best(10_000, -4.0);
        // Four histories start at 50_000. An earlier quiet stretch is not
        // a plateau, including one that would fit a shorter budget.
        assert!(!control.due(25_001));
        assert!(control.due(50_001));
    }

    #[test]
    fn acceptance_above_the_band_grows_the_step_and_a_half_does_not() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        for _ in 0..WINDOW {
            control.observe_trial(true);
        }
        assert!((control.step() - 0.38 * GROW).abs() < 1e-12);
        assert!((control.temperature() - 0.8).abs() < 1e-12);
        let mut even = WalkControl::new(0.38, 1.0, 0.8);
        for i in 0..WINDOW {
            even.observe_trial(i % 2 == 0);
        }
        assert!((even.step() - 0.38).abs() < 1e-12);
        let mut cold = WalkControl::new(0.38, 1.0, 0.8);
        for _ in 0..WINDOW {
            cold.observe_trial(false);
        }
        assert!((cold.step() - 0.38 / GROW).abs() < 1e-12);
    }

    #[test]
    fn a_step_against_its_bound_moves_the_temperature() {
        let mut control = WalkControl::new(STEP_HI, 1.0, 0.8);
        for _ in 0..WINDOW {
            control.observe_trial(true);
        }
        assert!((control.step() - STEP_HI).abs() < 1e-12);
        assert!((control.temperature() - 0.8 / GROW).abs() < 1e-12);
        let mut floored = WalkControl::new(STEP_LO, 1.0, 0.8);
        for _ in 0..WINDOW {
            floored.observe_trial(false);
        }
        assert!((floored.step() - STEP_LO).abs() < 1e-12);
        assert!((floored.temperature() - 0.8 * GROW).abs() < 1e-12);
    }

    #[test]
    fn the_second_failure_restarts_from_the_population() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        let incumbent = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let other = Array1::from(vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
        let incumbent_fp = fingerprint(incumbent.view(), 1.0);
        let other_fp = fingerprint(other.view(), 1.0);
        assert_ne!(incumbent_fp, other_fp);
        control.remember(incumbent_fp, -2.0, incumbent.view());
        control.remember(other_fp, -1.0, other.view());
        assert!(matches!(control.opening(incumbent_fp), Opening::Incumbent));
        control.note_failed(10, 5);
        assert!(matches!(control.opening(incumbent_fp), Opening::Random));
        control.note_failed(20, 5);
        match control.opening(incumbent_fp) {
            Opening::Stored { energy, coords } => {
                assert!((energy + 1.0).abs() < 1e-12);
                assert_eq!(coords[3], 3.0);
            }
            _ => panic!("a stored member is the third opening"),
        }
    }

    #[test]
    fn a_repeated_fingerprint_is_one_member() {
        let mut control = WalkControl::new(0.38, 1.0, 0.8);
        let coords = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let fp = fingerprint(coords.view(), 1.0);
        control.remember(fp, -1.0, coords.view());
        control.remember(fp, -1.5, coords.view());
        assert_eq!(control.members.len(), 1);
        assert!((control.members[0].energy + 1.5).abs() < 1e-12);
        assert!(!control.forbids(fp));
        control.note_avoid(fp);
        assert!(control.forbids(fp));
    }
}
