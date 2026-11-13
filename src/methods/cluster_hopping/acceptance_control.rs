//! Step, temperature, move choice, and a fingerprint population, from the run.
//!
//! Nothing here is a size or a reference energy. The step and the acceptance
//! temperature move when the recent acceptance is off a half. Each move is
//! drawn from the successes and failures it has produced. A basin fingerprint
//! enters the population once, and a run of fingerprints the population
//! already holds is a plateau: the chain restarts from a member it has not
//! been standing on.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

const WINDOW: usize = 32;
const TARGET: f64 = 0.5;
const FACTOR: f64 = 1.05;
const STEP_MIN: f64 = 0.25;
const STEP_MAX: f64 = 4.0;
const TEMP_MIN: f64 = 0.25;
const TEMP_MAX: f64 = 8.0;
/// Distinct fingerprints kept. A larger bank would only repeat the same
/// deduplicated set.
const POPULATION_CAP: usize = 24;

struct Member {
    basin: u64,
    energy: f64,
    coords: Array1<f64>,
    visits: usize,
}

/// A fingerprint of the pair-distance histogram, in units of `length`.
///
/// One pass over the pairs. The same geometry hashes the same way, and a
/// different packing lands in different bins. It does not consult the basin
/// index, so filing a structure does not scan every minimum the run has seen.
pub fn fingerprint(coords: ArrayView1<f64>, length: f64) -> u64 {
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

/// The four controls, inert until the search records what happened.
pub struct AcceptanceControl {
    step_scale: f64,
    temperature_scale: f64,
    window_n: usize,
    window_yes: usize,
    success: Vec<f64>,
    failure: Vec<f64>,
    members: Vec<Member>,
    streak: usize,
    spent_at_novel: usize,
    restarts: usize,
}

impl AcceptanceControl {
    /// Scales of one, an empty population, and no evidence on any move.
    pub fn new() -> Self {
        Self {
            step_scale: 1.0,
            temperature_scale: 1.0,
            window_n: 0,
            window_yes: 0,
            success: Vec::new(),
            failure: Vec::new(),
            members: Vec::new(),
            streak: 0,
            spent_at_novel: 0,
            restarts: 0,
        }
    }

    /// Multiplier on the proposal amplitude. One until the first window closes.
    pub fn step_scale(&self) -> f64 {
        self.step_scale
    }

    /// Multiplier on the acceptance temperature. One until the first window closes.
    pub fn temperature_scale(&self) -> f64 {
        self.temperature_scale
    }

    /// One proposal's acceptance, and a closed window retunes both scales.
    pub fn observe_acceptance(&mut self, accepted: bool) {
        self.window_n += 1;
        if accepted {
            self.window_yes += 1;
        }
        if self.window_n < WINDOW {
            return;
        }
        let rate = self.window_yes as f64 / self.window_n as f64;
        if rate > TARGET {
            self.step_scale = (self.step_scale * FACTOR).min(STEP_MAX);
            self.temperature_scale = (self.temperature_scale / FACTOR).max(TEMP_MIN);
        } else if rate < TARGET {
            self.step_scale = (self.step_scale / FACTOR).max(STEP_MIN);
            self.temperature_scale = (self.temperature_scale * FACTOR).min(TEMP_MAX);
        }
        self.window_n = 0;
        self.window_yes = 0;
    }

    /// Draws a move by Thompson sampling on its success and failure counts.
    pub fn select_arm<R: Rng + ?Sized>(&mut self, n_arms: usize, rng: &mut R) -> usize {
        self.ensure_arms(n_arms);
        let mut best = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for arm in 0..n_arms {
            let draw = beta_draw(1.0 + self.success[arm], 1.0 + self.failure[arm], rng);
            if draw > best_v {
                best_v = draw;
                best = arm;
            }
        }
        best
    }

    /// Records whether the move that was drawn produced an accepted improvement
    /// or a new basin the chain kept.
    pub fn observe_arm(&mut self, arm: usize, success: bool) {
        self.ensure_arms(arm.saturating_add(1));
        if success {
            self.success[arm] += 1.0;
        } else {
            self.failure[arm] += 1.0;
        }
    }

    /// Files a quenched basin once. A repeat lengthens the duplicate streak
    /// and returns false.
    pub fn observe_structure(
        &mut self,
        basin: u64,
        energy: f64,
        coords: ArrayView1<f64>,
        spent: usize,
    ) -> bool {
        if let Some(member) = self.members.iter_mut().find(|member| member.basin == basin) {
            member.visits += 1;
            if energy < member.energy {
                member.energy = energy;
                member.coords = coords.to_owned();
            }
            self.streak += 1;
            return false;
        }
        self.streak = 0;
        self.spent_at_novel = spent;
        self.members.push(Member {
            basin,
            energy,
            coords: coords.to_owned(),
            visits: 1,
        });
        self.shrink_population();
        true
    }

    /// True when the population has stopped admitting fingerprints and the
    /// best has been quiet for `patience` charges.
    pub fn restart_due(&self, spent: usize, patience: usize) -> bool {
        let bar = self.members.len().max(8);
        self.streak >= bar && spent.saturating_sub(self.spent_at_novel) >= patience.max(1)
    }

    /// Forgets the duplicate run so the next plateau has to earn another restart.
    pub fn acknowledge_restart(&mut self, spent: usize) {
        self.streak = 0;
        self.spent_at_novel = spent;
        self.restarts += 1;
    }

    /// The record has not moved. Raise the acceptance temperature so a worse
    /// minimum can be kept. The step is left to the acceptance window.
    pub fn on_quiet_best(&mut self) {
        self.temperature_scale = (self.temperature_scale * FACTOR * FACTOR).min(TEMP_MAX);
        self.window_n = 0;
        self.window_yes = 0;
    }

    /// A stored minimum whose fingerprint is not `current`, preferring one
    /// the chain has used least.
    pub fn diverse_member(&mut self, current: u64) -> Option<(f64, Array1<f64>)> {
        let mut pick: Option<usize> = None;
        for (index, member) in self.members.iter().enumerate() {
            if member.basin == current {
                continue;
            }
            let take = match pick {
                None => true,
                Some(held) => {
                    let other = &self.members[held];
                    member.visits < other.visits
                        || (member.visits == other.visits && member.energy < other.energy)
                }
            };
            if take {
                pick = Some(index);
            }
        }
        let index = pick?;
        self.members[index].visits += 1;
        Some((
            self.members[index].energy,
            self.members[index].coords.clone(),
        ))
    }

    fn ensure_arms(&mut self, n_arms: usize) {
        if self.success.len() < n_arms {
            self.success.resize(n_arms, 0.0);
            self.failure.resize(n_arms, 0.0);
        }
    }

    fn shrink_population(&mut self) {
        while self.members.len() > POPULATION_CAP {
            let best = self
                .members
                .iter()
                .map(|member| member.energy)
                .fold(f64::INFINITY, f64::min);
            let drop = self
                .members
                .iter()
                .enumerate()
                .filter(|(_, member)| member.energy > best)
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

fn beta_draw<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let left = sample_gamma(alpha, rng);
    let right = sample_gamma(beta, rng);
    let total = left + right;
    if total > 0.0 { left / total } else { 0.5 }
}

fn sample_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        let u: f64 = rng.random::<f64>().max(1e-300);
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u: f64 = rng.random::<f64>().max(1e-300);
        if u.ln() < 0.5 * z * z + d - d * v3 + d * v3.ln() {
            return d * v3;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn the_scales_start_at_one_and_follow_the_acceptance() {
        let mut control = AcceptanceControl::new();
        assert_eq!(control.step_scale(), 1.0);
        assert_eq!(control.temperature_scale(), 1.0);
        for _ in 0..WINDOW - 1 {
            control.observe_acceptance(true);
        }
        assert_eq!(control.step_scale(), 1.0);
        control.observe_acceptance(true);
        assert!((control.step_scale() - FACTOR).abs() < 1e-12);
        assert!((control.temperature_scale() - 1.0 / FACTOR).abs() < 1e-12);

        let mut cold = AcceptanceControl::new();
        for _ in 0..WINDOW {
            cold.observe_acceptance(false);
        }
        assert!((cold.step_scale() - 1.0 / FACTOR).abs() < 1e-12);
        assert!((cold.temperature_scale() - FACTOR).abs() < 1e-12);
    }

    #[test]
    fn an_acceptance_of_one_half_leaves_both_scales_alone() {
        let mut control = AcceptanceControl::new();
        for i in 0..WINDOW {
            control.observe_acceptance(i % 2 == 0);
        }
        assert_eq!(control.step_scale(), 1.0);
        assert_eq!(control.temperature_scale(), 1.0);
    }

    #[test]
    fn a_repeated_fingerprint_is_not_a_second_member() {
        let mut control = AcceptanceControl::new();
        let coords = Array1::from(vec![0.0, 0.0, 0.0]);
        assert!(control.observe_structure(3, -1.0, coords.view(), 10));
        assert!(!control.observe_structure(3, -1.0, coords.view(), 20));
        assert_eq!(control.members.len(), 1);
        assert_eq!(control.members[0].visits, 2);
        assert!(control.observe_structure(4, -2.0, coords.view(), 30));
        assert_eq!(control.members.len(), 2);
        assert!(!control.restart_due(30, 100));
    }

    #[test]
    fn a_duplicate_run_with_a_quiet_best_is_a_plateau() {
        let mut control = AcceptanceControl::new();
        let coords = Array1::from(vec![0.0, 0.0, 0.0]);
        control.observe_structure(1, -1.0, coords.view(), 0);
        for spent in 1..9 {
            control.observe_structure(1, -1.0, coords.view(), spent);
        }
        assert!(!control.restart_due(8, 100));
        assert!(control.restart_due(100, 100));
        let other = Array1::from(vec![1.0, 0.0, 0.0]);
        control.observe_structure(2, -0.5, other.view(), 100);
        let (energy, state) = control.diverse_member(1).expect("a second fingerprint");
        assert!((energy + 0.5).abs() < 1e-12);
        assert_eq!(state[0], 1.0);
        control.acknowledge_restart(100);
        assert!(!control.restart_due(100, 100));
    }

    #[test]
    fn the_arm_that_succeeds_is_drawn_more_often() {
        let mut control = AcceptanceControl::new();
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..40 {
            control.observe_arm(0, true);
            control.observe_arm(1, false);
        }
        let mut counts = [0usize; 2];
        for _ in 0..200 {
            counts[control.select_arm(2, &mut rng)] += 1;
        }
        assert!(
            counts[0] > counts[1],
            "draws were {counts:?}"
        );
    }
}
