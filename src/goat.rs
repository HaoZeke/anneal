//! Search rules of GOAT, and the CREST criteria it sorts minima by.
//!
//! GOAT is the global optimization algorithm of B. de Souza, *GOAT: A
//! Global Optimization Algorithm for Molecules and Atomic Clusters*,
//! Angew. Chem. Int. Ed. **64**, e202500393 (2025),
//! <https://doi.org/10.1002/anie.202500393>. Its workers walk from minimum
//! to minimum under a Metropolis test at temperatures that regulate
//! themselves.
//!
//! Only the rules are carried here, and each item names the paper its rule
//! and each of its numbers come from. Where the rule here departs from
//! the paper, the item says how and why.
//!
//! GOAT states its energies in kcal/mol. A function that takes one reads
//! it in a unit where one kcal/mol is `kcal_per_mol`, and
//! [`Config::kcal_per_mol`] gives that size for a run. A run in reduced
//! units has no chemical unit, so there a kcal/mol figure is read as the
//! same multiple of [`Config::energy_scale`].
//!
//! [`Config::kcal_per_mol`]: crate::methods::cluster_hopping::Config::kcal_per_mol
//! [`Config::energy_scale`]: crate::methods::cluster_hopping::Config::energy_scale

/// Uphill steps, in kcal/mol, that GOAT's four base workers accept half
/// the time (de Souza 2025).
///
/// Each step doubles the one before, so the base temperatures form a
/// geometric ladder of ratio two whose top is eight times its bottom.
pub const LADDER_STEPS_KCAL: [f64; 4] = [0.5, 1.0, 2.0, 4.0];

/// Temperature, in the unit of `step`, at which the Metropolis test takes
/// a rise of `step` with probability one half.
///
/// GOAT's eq. (2) takes a rise `dE` with probability `exp(-dE / k_B T)`.
/// That probability is one half at `dE = step` when `k_B T = step / ln 2`.
pub fn half_acceptance_temperature(step: f64) -> f64 {
    step / std::f64::consts::LN_2
}

/// Base temperatures of GOAT's four workers, coldest first, in a unit
/// where one kcal/mol is `kcal_per_mol`.
pub fn ladder_temperatures(kcal_per_mol: f64) -> [f64; 4] {
    LADDER_STEPS_KCAL.map(|step| half_acceptance_temperature(step * kcal_per_mol))
}

/// Factor by which a walker's temperature moves after each outcome.
///
/// GOAT's default `T_fac` (de Souza 2025). A taken step divides the
/// temperature by it and a refused one multiplies it, and the temperature
/// never falls below the walker's base. GOAT moves its uphill limit
/// `MaxEn` by the same factor; the hop here has no uphill walk, so only
/// the temperature is carried.
///
/// The factor sets the speed of the feedback and not its target. Let a
/// walker's level be `u = ln(T / T_base) / ln(T_fac)`. Each outcome moves
/// `u` down by one with probability `p(T)`, the chance that a step is
/// taken at `T`, and up by one otherwise. The mean drift `1 - 2 p(T)`
/// vanishes where `p(T) = 1/2`, the acceptance the base temperatures were
/// built for, whatever the factor.
///
/// What the factor fixes is how far one walker can wander toward the
/// next. The ladder doubles from rung to rung, which is
/// `ln 2 / ln 1.05 = 14.2` levels. Without drift the level spreads as the
/// square root of the number of outcomes, so a walker needs about
/// `14.2^2 = 202` outcomes to wander one rung, and the drift toward
/// `p = 1/2` only shortens such excursions.
pub const TEMPERATURE_FACTOR: f64 = 1.05;

/// A walker's temperature under GOAT's feedback, as a factor on its base.
///
/// The factor is `TEMPERATURE_FACTOR^level` for a level of at least zero.
/// The level is an integer so that the floor is exact: a long run of taken
/// steps brings the walker back to its base and not to a rounding error
/// below it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelfRegulatingTemperature {
    level: u32,
}

impl SelfRegulatingTemperature {
    /// A walker at its base temperature.
    pub fn new() -> Self {
        Self::default()
    }

    /// Factor on the base temperature, at least one.
    pub fn factor(&self) -> f64 {
        TEMPERATURE_FACTOR.powi(self.level.min(i32::MAX as u32) as i32)
    }

    /// Feedback steps above the base.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Applies one Metropolis outcome between two distinct minima.
    ///
    /// GOAT counts a duplicate minimum as a refused structure and leaves
    /// the temperature alone, so the caller passes only trials that reached
    /// another minimum.
    pub fn observe(&mut self, accepted: bool) {
        self.level = if accepted {
            self.level.saturating_sub(1)
        } else {
            self.level.saturating_add(1)
        };
    }
}

/// Length of the longest run of misses that `trials` independent trials,
/// each a hit with probability `p`, show about once.
///
/// A run of at least `L` misses starts after a hit, so `trials` trials
/// contain about `trials * p * (1 - p)^L` of them. That count is one at
/// `L = ln(trials * p) / ln(1 / (1 - p))`, the leading term of the longest
/// run in M. F. Schilling, College Math. J. **21**, 196 (1990). The value
/// is zero when there is nothing to measure: `p` at zero or one, or fewer
/// than one expected hit.
pub fn expected_longest_run(trials: u64, p: f64) -> f64 {
    let hits = trials as f64 * p;
    if !(p > 0.0 && p < 1.0) || hits <= 1.0 {
        return 0.0;
    }
    hits.ln() / (1.0 / (1.0 - p)).ln()
}

/// GOAT's gradient boost reduced to its forced step: after a stall, take
/// the next new minimum whatever its energy.
///
/// GOAT enters the boost when no new minimum turns up for more than
/// `GradBoost` iterations, three by default, and then takes the next new
/// minimum regardless of the temperature (de Souza 2025). It also
/// strengthens its uphill gradient, and the hop here has no uphill walk,
/// so only the forced step is carried.
///
/// The stall length is not GOAT's. Three counts GOAT micro-iterations,
/// each a full uphill walk and relaxation that finds a new minimum most of
/// the time. A hop finds one at a rate that differs by system and falls as
/// the bias fills, so the stall is measured against the walker's own rate
/// instead: the walker is stalled once its current run without a new
/// minimum reaches [`expected_longest_run`] at its measured discovery
/// rate. A walk whose rate stays the same reaches that length about once
/// in its history so far. At trial `t` such a run ends with probability
/// about `1 / t`, so the trials from `H` to `2H` contain about `ln 2` of
/// them, and a steady walk is boosted about 0.69 times or fewer each time
/// its history doubles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StallBoost {
    trials: u64,
    new: u64,
    run: u64,
    armed: bool,
}

impl StallBoost {
    /// A walker with no history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one quenched trial, `new` when it reached a minimum the
    /// walker had not stood in.
    pub fn observe(&mut self, new: bool) {
        self.trials += 1;
        if new {
            self.new += 1;
            self.run = 0;
        } else {
            self.run += 1;
        }
        let limit = expected_longest_run(self.trials, self.new as f64 / self.trials as f64);
        self.armed |= limit > 0.0 && self.run as f64 >= limit;
    }

    /// Whether the next new minimum is to be taken unconditionally.
    pub fn armed(&self) -> bool {
        self.armed
    }

    /// Spends the boost on a new minimum.
    pub fn fire(&mut self) {
        self.armed = false;
    }

    /// Current run of trials without a new minimum.
    pub fn run(&self) -> u64 {
        self.run
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_temperatures_take_their_step_half_the_time() {
        for kcal in [1.0, 4.184, 0.37] {
            for (step, t) in LADDER_STEPS_KCAL.iter().zip(ladder_temperatures(kcal)) {
                let p = (-(step * kcal) / t).exp();
                assert!((p - 0.5).abs() < 1e-12, "step {step} taken with {p}");
            }
        }
    }

    #[test]
    fn a_kilojoule_ladder_is_4_184_times_the_kilocalorie_one() {
        let kcal = ladder_temperatures(1.0);
        let kj = ladder_temperatures(4.184);
        for (a, b) in kcal.iter().zip(kj.iter()) {
            assert!((b / a - 4.184).abs() < 1e-12);
        }
        assert!((kj[0] - 0.5 * 4.184 / std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn the_ladder_doubles_from_rung_to_rung() {
        let t = ladder_temperatures(1.0);
        for k in 1..t.len() {
            assert!((t[k] / t[k - 1] - 2.0).abs() < 1e-12);
        }
        assert!((t[3] / t[0] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn feedback_never_cools_below_the_base() {
        let mut walker = SelfRegulatingTemperature::new();
        for _ in 0..10 {
            walker.observe(true);
        }
        assert_eq!(walker.level(), 0);
        assert_eq!(walker.factor(), 1.0);
    }

    #[test]
    fn one_refusal_and_one_acceptance_cancel() {
        let mut walker = SelfRegulatingTemperature::new();
        walker.observe(false);
        assert!((walker.factor() - TEMPERATURE_FACTOR).abs() < 1e-15);
        walker.observe(false);
        walker.observe(true);
        assert!((walker.factor() - TEMPERATURE_FACTOR).abs() < 1e-15);
        walker.observe(true);
        assert_eq!(walker.factor(), 1.0);
    }

    /// Uniform numbers in `[0, 1)` from a xorshift generator.
    fn uniforms(mut state: u64) -> impl FnMut() -> f64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn feedback_settles_where_half_the_steps_are_taken() {
        // A walker whose acceptance falls with temperature exactly as a
        // Metropolis rise of `step` does, started four times too cold.
        let step = 1.0;
        let base = 0.25 * half_acceptance_temperature(step);
        let mut walker = SelfRegulatingTemperature::new();
        let mut uniform = uniforms(0x2545_f491_4f6c_dd1d);
        let (mut taken, mut counted) = (0usize, 0usize);
        for k in 0..200_000 {
            let accept = uniform() < (-step / (base * walker.factor())).exp();
            if k >= 20_000 {
                taken += usize::from(accept);
                counted += 1;
            }
            walker.observe(accept);
        }
        let rate = taken as f64 / counted as f64;
        assert!((rate - 0.5).abs() < 0.01, "settled at {rate}");
    }

    #[test]
    fn the_longest_run_is_the_one_expected_once() {
        let (h, p) = (10_000u64, 0.2);
        let l = expected_longest_run(h, p);
        let count = h as f64 * p * (1.0 - p).powf(l);
        assert!((count - 1.0).abs() < 1e-9, "count {count}");
        assert_eq!(expected_longest_run(h, 0.0), 0.0);
        assert_eq!(expected_longest_run(h, 1.0), 0.0);
        assert_eq!(expected_longest_run(3, 0.2), 0.0);
    }

    #[test]
    fn a_steady_walk_is_boosted_less_than_once_per_doubling() {
        let p = 0.3;
        let mut uniform = uniforms(0x9e37_79b9_7f4a_7c15);
        let walks = 200;
        let mut fired = 0u32;
        for _ in 0..walks {
            let mut boost = StallBoost::new();
            for _ in 0..1u64 << 16 {
                let new = uniform() < p;
                boost.observe(new);
                if new && boost.armed() {
                    boost.fire();
                    fired += 1;
                }
            }
        }
        let per_doubling = fired as f64 / walks as f64 / 16.0;
        assert!(per_doubling > 0.1 && per_doubling < std::f64::consts::LN_2);
    }

    #[test]
    fn a_stalled_walk_is_boosted() {
        let mut boost = StallBoost::new();
        for k in 0..1000 {
            let new = k % 2 == 0;
            boost.observe(new);
            if new && boost.armed() {
                boost.fire();
            }
        }
        assert!(!boost.armed());
        let mut waited = 0;
        while !boost.armed() {
            boost.observe(false);
            waited += 1;
        }
        // The warm-up ends on a miss, so the run starts at one.
        let limit = expected_longest_run(boost.trials, boost.new as f64 / boost.trials as f64);
        assert_eq!(waited as f64 + 1.0, limit.ceil());
        boost.observe(true);
        assert!(boost.armed());
        boost.fire();
        assert!(!boost.armed());
    }
}
