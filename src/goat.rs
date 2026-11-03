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
}
