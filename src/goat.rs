//! Search rules of GOAT, and the CREST criteria it sorts minima by.
//!
//! GOAT is the global optimization algorithm of B. de Souza, *GOAT: A
//! Global Optimization Algorithm for Molecules and Atomic Clusters*,
//! Angew. Chem. Int. Ed. **64**, e202500393 (2025),
//! <https://doi.org/10.1002/anie.202500393>. Its workers walk from minimum
//! to minimum under a Metropolis test at temperatures that regulate
//! themselves.
//! It sorts the minima it finds by energy and rotational constants, with
//! criteria it takes from CREST: P. Pracht, F. Bohle and S. Grimme, Phys.
//! Chem. Chem. Phys. **22**, 7169 (2020), whose default thresholds are
//! tabulated in P. Pracht *et al.*, J. Chem. Phys. **160**, 114110 (2024).
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

/// Energy window, in kcal/mol, inside which two minima can be rotamers.
///
/// CREST's default `E_thr` (Pracht *et al.* 2024, Table II). GOAT's own
/// default `MaxEnDiff` is 0.1 kcal/mol (de Souza 2025); the narrower CREST
/// window is used here.
pub const ROTAMER_WINDOW_KCAL: f64 = 0.05;

/// Largest relative difference of the rotational constants of two rotamers
/// of a spherical top.
///
/// GOAT's `MaxRotDiff` of at most one percent (de Souza 2025), which is
/// also the low end of CREST's `B_thr` (Pracht *et al.* 2024, Table II).
pub const ROTOR_TOLERANCE_SPHERICAL: f64 = 0.01;

/// The same tolerance for a rotor whose smallest moment of inertia is
/// negligible beside its largest.
///
/// The high end of CREST's `B_thr` range of 1.0 to 2.5 percent (Pracht
/// *et al.* 2024, Table II).
pub const ROTOR_TOLERANCE_ANISOTROPIC: f64 = 0.025;

/// Standard atomic weight of element `z`, in daltons.
///
/// Abridged values of the IUPAC Commission on Isotopic Abundances and
/// Atomic Weights (Prohaska *et al.*, Pure Appl. Chem. **94**, 573 (2022))
/// for hydrogen to krypton and for the coinage and platinum-group metals.
/// Only ratios of moments of inertia are used, so the unit does not
/// matter.
pub fn standard_atomic_weight(z: u32) -> Option<f64> {
    const LIGHT: [f64; 36] = [
        1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305,
        26.982, 28.085, 30.974, 32.06, 35.45, 39.95, 39.098, 40.078, 44.956, 47.867, 50.942,
        51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971,
        79.904, 83.798,
    ];
    match z {
        1..=36 => Some(LIGHT[z as usize - 1]),
        44 => Some(101.07),
        45 => Some(102.91),
        46 => Some(106.42),
        47 => Some(107.87),
        76 => Some(190.23),
        77 => Some(192.22),
        78 => Some(195.08),
        79 => Some(196.97),
        _ => None,
    }
}

/// Principal moments of inertia of a structure, smallest first.
///
/// A rotational constant is `B = h / (8 pi^2 I)`, so the relative gap
/// `|B - B'| / max(B, B')` between two constants equals the relative gap
/// `|I - I'| / max(I, I')` between their moments. The test of GOAT and
/// CREST on the constants is therefore made on the moments, and no unit
/// of `B` enters.
///
/// Atom permutations leave the moments unchanged, and that lets the
/// test join rotamers that a descriptor or an aligned distance can miss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotor {
    moments: [f64; 3],
}

impl Rotor {
    /// The rotor of `points`, three coordinates per site, with site masses
    /// `masses`.
    ///
    /// The mass-weighted second-moment tensor `S = sum m d d^T` about the
    /// centre of mass has the principal values `s_k`, and the inertia
    /// tensor `tr(S) - S` shares its eigenvectors, so the moments are
    /// `I_k = tr(S) - s_k`. None when the lengths disagree or a moment is
    /// not positive, as for a single site or sites on one line.
    pub fn of(points: ndarray::ArrayView1<f64>, masses: &[f64]) -> Option<Self> {
        let n = masses.len();
        if n < 2 || points.len() != 3 * n {
            return None;
        }
        let total: f64 = masses.iter().sum();
        if !(total > 0.0) {
            return None;
        }
        let mut centre = [0.0_f64; 3];
        for (i, m) in masses.iter().enumerate() {
            for (k, c) in centre.iter_mut().enumerate() {
                *c += m * points[3 * i + k] / total;
            }
        }
        let mut second = [[0.0_f64; 3]; 3];
        for (i, m) in masses.iter().enumerate() {
            let d: [f64; 3] = std::array::from_fn(|k| points[3 * i + k] - centre[k]);
            for (a, row) in second.iter_mut().enumerate() {
                for (b, entry) in row.iter_mut().enumerate() {
                    *entry += m * d[a] * d[b];
                }
            }
        }
        let (principal, _) = crate::methods::two_phase::symmetric_eigen(second);
        let trace: f64 = principal.iter().sum();
        let mut moments = principal.map(|s| trace - s);
        moments.sort_by(f64::total_cmp);
        let floor = 1e-12 * moments[2].abs().max(f64::MIN_POSITIVE);
        moments
            .iter()
            .all(|m| m.is_finite() && *m > floor)
            .then_some(Self { moments })
    }

    /// Principal moments, smallest first.
    pub fn moments(&self) -> [f64; 3] {
        self.moments
    }

    /// `1 - I_min / I_max`: zero for a spherical top and near one when the
    /// smallest moment is negligible beside the largest.
    pub fn anisotropy(&self) -> f64 {
        (1.0 - self.moments[0] / self.moments[2]).clamp(0.0, 1.0)
    }

    /// Tolerance on the relative difference of this rotor's constants.
    ///
    /// CREST widens its `B_thr` from 1.0 to 2.5 percent as a rotor grows
    /// more anisotropic (Pracht *et al.* 2024, Table II). The paper gives
    /// the range and the direction and not the curve. The curve here is
    /// anneal's: a straight line in [`Rotor::anisotropy`] from
    /// [`ROTOR_TOLERANCE_SPHERICAL`] to [`ROTOR_TOLERANCE_ANISOTROPIC`].
    pub fn tolerance(&self) -> f64 {
        ROTOR_TOLERANCE_SPHERICAL
            + (ROTOR_TOLERANCE_ANISOTROPIC - ROTOR_TOLERANCE_SPHERICAL) * self.anisotropy()
    }

    /// Largest relative difference between matching principal moments.
    pub fn difference(&self, other: &Self) -> f64 {
        self.moments
            .iter()
            .zip(other.moments.iter())
            .map(|(a, b)| (a - b).abs() / a.max(*b))
            .fold(0.0, f64::max)
    }

    /// Whether two rotors pass the rotational-constant test of GOAT's
    /// third sorting step (de Souza 2025).
    ///
    /// The test is symmetric: the pair must pass the wider of its two
    /// tolerances. The energy test is the caller's.
    pub fn matches(&self, other: &Self) -> bool {
        self.difference(other) < self.tolerance().max(other.tolerance())
    }
}

/// Minima kept for the rotamer test, each with a value the caller maps a
/// rotamer onto.
///
/// GOAT's sorting first takes every kept minimum within the energy window,
/// and calls a new minimum the same as one of those whose rotational
/// constants agree within the tolerance (de Souza 2025, steps 1 and 3).
/// GOAT's step 2, an aligned distance below 0.125 angstrom, is not
/// carried: the cluster hop already merges minima by its own descriptors.
#[derive(Debug, Clone)]
pub struct RotamerArchive<T> {
    entries: Vec<(f64, Rotor, T)>,
    window: f64,
}

impl<T> RotamerArchive<T> {
    /// An empty archive whose energy test is `|E_a - E_b| < window`.
    pub fn new(window: f64) -> Self {
        Self {
            entries: Vec::new(),
            window,
        }
    }

    /// The value of the first kept minimum that a minimum at `energy` with
    /// rotor `rotor` is a rotamer of.
    pub fn find(&self, energy: f64, rotor: &Rotor) -> Option<&T> {
        self.entries
            .iter()
            .find(|(e, kept, _)| (e - energy).abs() < self.window && kept.matches(rotor))
            .map(|(_, _, value)| value)
    }

    /// Keeps a minimum.
    pub fn push(&mut self, energy: f64, rotor: Rotor, value: T) {
        self.entries.push((energy, rotor, value));
    }

    /// Kept minima.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is kept.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Molar gas constant in kcal/(mol K), from the exact CODATA 2018 value
/// of 8.314462618 J/(mol K) and 4.184 J per calorie.
const GAS_CONSTANT_KCAL: f64 = 8.314_462_618e-3 / 4.184;

/// Temperature of GOAT's conformer ensemble, in kelvin.
///
/// The paper reports its ensemble entropies and free energies at 298.15 K
/// (de Souza 2025), and ORCA's `CONFTEMP` defaults to it (ORCA 6.1
/// manual, GOAT keyword list).
pub const CONFORMER_KELVIN: f64 = 298.15;

/// Highest energy of a conformer above the lowest, in kcal/mol.
///
/// ORCA's `MAXEN` default for GOAT (ORCA 6.1 manual, GOAT keyword list).
pub const MAXEN_KCAL: f64 = 6.0;

/// [`MAXEN_KCAL`] in units of `k_B T` at [`CONFORMER_KELVIN`], about 10.13.
///
/// A member at the edge carries a Boltzmann factor of about 4e-5 against
/// the lowest. Stated in `k_B T`, the window keeps that meaning in a run
/// that has no kelvin, one in reduced units.
pub const ENSEMBLE_WINDOW: f64 = MAXEN_KCAL / (GAS_CONSTANT_KCAL * CONFORMER_KELVIN);

/// Change of the ensemble entropy between rounds below which the entropy
/// counts as converged, in units of `k_B`, about 0.0503.
///
/// ORCA's `MINDELS` default of 0.1 cal/(mol K) for GOAT-ENTROPY (ORCA 6.1
/// manual), divided by the gas constant.
pub const ENTROPY_TOLERANCE: f64 = 0.1e-3 / GAS_CONSTANT_KCAL;

/// Rounds a run completes before it may stop.
///
/// GOAT runs at least three global cycles by default (de Souza 2025),
/// ORCA's `MINGLOBALITER`.
pub const MIN_ROUNDS: usize = 3;

/// `k_B T` at `kelvin`, in a unit where one kcal/mol is `kcal_per_mol`.
pub fn thermal_energy(kelvin: f64, kcal_per_mol: f64) -> f64 {
    GAS_CONSTANT_KCAL * kelvin * kcal_per_mol
}

/// GOAT's optimisations per base worker and global cycle for `atoms`
/// atoms in `fragments` fragments.
///
/// This is eq. (4) of de Souza 2025, `max(f_flex N^(3/2), N, 15)` with
/// `N` the atom count. With no rotatable bonds, as in a water cluster, the
/// paper sets `f_flex = 0.5 sqrt(N_frag)`. A chart here moves atoms and
/// rigid bodies and has no rotatable bond, so that value applies to every
/// chart, and the fragment count is the caller's.
pub fn optimizations_per_worker(atoms: usize, fragments: usize) -> usize {
    let flex = 0.5 * (fragments.max(1) as f64).sqrt();
    let scaled = (flex * (atoms as f64).powf(1.5)).ceil() as usize;
    scaled.max(atoms).max(15)
}

/// Quenched minima in one round of the entropy stop.
///
/// A GOAT global cycle runs [`optimizations_per_worker`] optimisations on
/// each of its four base workers, however many workers the base is split
/// into (de Souza 2025). A round here is that work, by however many chains
/// share it.
pub fn round_length(atoms: usize, fragments: usize) -> usize {
    LADDER_STEPS_KCAL.len() * optimizations_per_worker(atoms, fragments)
}

/// Boltzmann summary of a conformer ensemble at one temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformerEnsemble {
    /// `k_B T` the weights are taken at, in energy units.
    pub temperature: f64,
    /// Unique minima within [`ENSEMBLE_WINDOW`] of the lowest.
    pub members: usize,
    /// Conformational entropy `-sum p_i ln p_i`, in units of `k_B`.
    pub entropy: f64,
    /// Conformational free energy `-k_B T ln sum exp(-dE_i / k_B T)`
    /// relative to the lowest member, in energy units.
    pub free_energy: f64,
    /// Rounds of the entropy stop completed.
    pub rounds: usize,
}

/// Member count, entropy and free energy of minima at `energies`, at
/// `k_B T = temperature`.
///
/// With every degeneracy one, as GOAT reports its ensembles (de Souza
/// 2025), the weights are `p_i = exp(-x_i) / Z` for `x_i = dE_i / k_B T`
/// and `Z = sum exp(-x_i)`. The entropy is then `S / k_B = ln Z + <x>`
/// and the free energy `G = -k_B T ln Z`. Minima more than
/// [`ENSEMBLE_WINDOW`] `k_B T` above the lowest are left out. Returns the
/// member count, `S / k_B` and `G` relative to the lowest member.
pub fn ensemble_entropy(energies: &[f64], temperature: f64) -> (usize, f64, f64) {
    let Some(lowest) = energies
        .iter()
        .copied()
        .filter(|e| e.is_finite())
        .reduce(f64::min)
    else {
        return (0, 0.0, 0.0);
    };
    if !(temperature > 0.0) {
        return (1, 0.0, 0.0);
    }
    let reduced: Vec<f64> = energies
        .iter()
        .map(|e| (e - lowest) / temperature)
        .filter(|x| x.is_finite() && *x <= ENSEMBLE_WINDOW)
        .collect();
    let z: f64 = reduced.iter().map(|x| (-x).exp()).sum();
    let mean = reduced.iter().map(|x| x * (-x).exp()).sum::<f64>() / z;
    (reduced.len(), z.ln() + mean, -temperature * z.ln())
}

/// GOAT-ENTROPY's stop over the rounds of a run.
///
/// GOAT stops after its minimum number of global cycles, at the end of a
/// cycle that found no lower minimum (de Souza 2025). GOAT-ENTROPY also
/// asks the conformational entropy to have moved by less than its
/// tolerance (ORCA 6.1 manual). The rounds here stand for the cycles, and
/// the stop takes both conditions after at least [`MIN_ROUNDS`] rounds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EntropyStop {
    rounds: usize,
    last: Option<(f64, f64)>,
}

impl EntropyStop {
    /// A stop that has seen no round.
    pub fn new() -> Self {
        Self::default()
    }

    /// Closes a round whose lowest energy so far is `best` and whose
    /// ensemble entropy is `entropy`, and says whether to stop.
    pub fn end_round(&mut self, best: f64, entropy: f64) -> bool {
        self.rounds += 1;
        let previous = self.last.replace((best, entropy));
        previous.is_some_and(|(last_best, last_entropy)| {
            self.rounds >= MIN_ROUNDS
                && best >= last_best
                && (entropy - last_entropy).abs() < ENTROPY_TOLERANCE
        })
    }

    /// Rounds closed so far.
    pub fn rounds(&self) -> usize {
        self.rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    fn water() -> (ndarray::Array1<f64>, [f64; 3]) {
        (
            array![0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0],
            [15.999, 1.008, 1.008],
        )
    }

    #[test]
    fn moments_of_a_planar_rotor_add_up() {
        let (points, masses) = water();
        let rotor = Rotor::of(points.view(), &masses).unwrap();
        let [a, b, c] = rotor.moments();
        assert!(a <= b && b <= c);
        assert!((a + b - c).abs() < 1e-9 * c, "{a} + {b} != {c}");
    }

    #[test]
    fn a_turned_relabelled_copy_is_a_rotamer_and_a_stretched_one_is_not() {
        let (points, masses) = water();
        let a = Rotor::of(points.view(), &masses).unwrap();
        // Turned by 90 degrees about z, with the hydrogens swapped.
        let turned = array![0.0, 0.0, 0.0, 0.586, 0.757, 0.0, 0.586, -0.757, 0.0];
        let b = Rotor::of(turned.view(), &masses).unwrap();
        assert!(a.difference(&b) < 1e-12);
        assert!(a.matches(&b));
        let mut stretched = points.clone();
        stretched[3] *= 1.05;
        stretched[6] *= 1.05;
        let c = Rotor::of(stretched.view(), &masses).unwrap();
        assert!(!a.matches(&c));
    }

    #[test]
    fn the_tolerance_runs_from_one_to_two_and_a_half_percent() {
        let cube: Vec<f64> = (0..8)
            .flat_map(|k| [k & 1, (k >> 1) & 1, (k >> 2) & 1].map(f64::from))
            .collect();
        let sphere = Rotor::of(ndarray::ArrayView1::from(&cube), &[1.0; 8]).unwrap();
        assert!(sphere.anisotropy() < 1e-12);
        assert!((sphere.tolerance() - ROTOR_TOLERANCE_SPHERICAL).abs() < 1e-12);
        let rod = array![0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.01, 0.0, 0.0];
        let rod = Rotor::of(rod.view(), &[1.0; 3]).unwrap();
        assert!(rod.anisotropy() > 0.999);
        assert!((rod.tolerance() - ROTOR_TOLERANCE_ANISOTROPIC).abs() < 1e-4);
        assert!(rod.tolerance() <= ROTOR_TOLERANCE_ANISOTROPIC);
    }

    #[test]
    fn collinear_sites_have_no_rotor() {
        let line = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert!(Rotor::of(line.view(), &[1.0; 3]).is_none());
        assert!(Rotor::of(line.view(), &[1.0; 2]).is_none());
    }

    #[test]
    fn the_archive_applies_the_energy_window() {
        let (points, masses) = water();
        let rotor = Rotor::of(points.view(), &masses).unwrap();
        let mut archive = RotamerArchive::new(ROTAMER_WINDOW_KCAL);
        archive.push(-1.0, rotor, 7usize);
        assert_eq!(archive.find(-1.0 + 0.04, &rotor), Some(&7));
        assert_eq!(archive.find(-1.0 + 0.06, &rotor), None);
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn weights_cover_water_and_platinum() {
        assert_eq!(standard_atomic_weight(1), Some(1.008));
        assert_eq!(standard_atomic_weight(8), Some(15.999));
        assert_eq!(standard_atomic_weight(78), Some(195.08));
        assert_eq!(standard_atomic_weight(0), None);
    }

    #[test]
    fn orca_constants_in_thermal_units() {
        assert!((ENSEMBLE_WINDOW - 10.13).abs() < 5e-3, "{ENSEMBLE_WINDOW}");
        assert!(
            (ENTROPY_TOLERANCE - 0.0503).abs() < 5e-5,
            "{ENTROPY_TOLERANCE}"
        );
        let room_kj = thermal_energy(CONFORMER_KELVIN, 4.184);
        assert!((room_kj - 2.4790).abs() < 1e-4, "{room_kj}");
    }

    #[test]
    fn rounds_follow_the_papers_iteration_count() {
        // A water decamer: 30 atoms in 10 fragments.
        assert_eq!(optimizations_per_worker(30, 10), 260);
        assert_eq!(round_length(30, 10), 1040);
        // Small systems fall back to the floor of 15 and to the atom count.
        assert_eq!(optimizations_per_worker(3, 1), 15);
        assert_eq!(optimizations_per_worker(4, 1), 15);
        assert_eq!(optimizations_per_worker(75, 75), 2813);
    }

    #[test]
    fn degenerate_minima_give_ln_n() {
        let (members, s, g) = ensemble_entropy(&[-1.0; 4], 0.5);
        assert_eq!(members, 4);
        assert!((s - 4.0_f64.ln()).abs() < 1e-12);
        assert!((g + 0.5 * 4.0_f64.ln()).abs() < 1e-12);
        let (members, s, _) = ensemble_entropy(&[0.0, 100.0], 1.0);
        assert_eq!(members, 1);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn two_levels_match_the_closed_form() {
        let (t, de) = (0.7, 0.3);
        let (_, s, g) = ensemble_entropy(&[0.0, de], t);
        let p = 1.0 / (1.0 + (-de / t).exp());
        let closed = -(p * p.ln() + (1.0 - p) * (1.0 - p).ln());
        assert!((s - closed).abs() < 1e-12);
        assert!((g + t * (1.0 + (-de / t).exp()).ln()).abs() < 1e-12);
    }

    #[test]
    fn the_stop_waits_three_rounds_and_a_quiet_one() {
        let mut stop = EntropyStop::new();
        assert!(!stop.end_round(-1.0, 1.0));
        assert!(!stop.end_round(-1.0, 1.0));
        assert!(stop.end_round(-1.0, 1.0 + 0.5 * ENTROPY_TOLERANCE));
        let mut moving = EntropyStop::new();
        assert!(!moving.end_round(-1.0, 1.0));
        assert!(!moving.end_round(-1.0, 1.1));
        assert!(!moving.end_round(-1.0, 1.2));
        assert!(!moving.end_round(-1.5, 1.2));
        assert!(moving.end_round(-1.5, 1.2));
        assert_eq!(moving.rounds(), 5);
    }
}
