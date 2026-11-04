//! Dimensionless coefficients for cluster-search presets.

/// Lennard-Jones coefficients multiplied by declared length and energy scales.
pub(super) struct LennardJonesPreset;

impl LennardJonesPreset {
    pub(super) const REDUCED_SCALE: f64 = 1.0;
    pub(super) const TEMPERATURE: f64 = 0.8;
    pub(super) const BIAS_HEIGHT: f64 = 0.25;
    pub(super) const MERGE_RADIUS: f64 = 0.7;
    pub(super) const ALL_POINTS_STEP: f64 = 0.38;
    pub(super) const SINGLE_POINT_STEP: f64 = 1.0;
    pub(super) const NEIGHBOUR_CUTOFF: f64 = 1.6;
    pub(super) const SYMMETRISE_CUTOFF: f64 = 2.5;
    pub(super) const SYMMETRY_MERGE_RADIUS: f64 = 0.7;
    pub(super) const GROUP_SHAKE: f64 = 0.3;
    pub(super) const GROUP_CUTOFF: f64 = 3.4;
    pub(super) const COVALENT_CUTOFF: f64 = 1.3;
    /// Measured, one value for every system rather than a per-system knob.
    /// At 0.35 the frozen arm reaches the LJ75 Marks decahedron in 20 of 48
    /// seeds and the LJ98 Leary tetrahedron in 21 of 48; at 0.5 LJ75 is 31 of
    /// 48, while LJ38 is unchanged at 70 of 72 and LJ55 at 48 of 48. Nothing
    /// measured pays for the looser match, so it is the default and no
    /// campaign sets `SYM_TOL`.
    pub(super) const SYMMETRY_TOLERANCE: f64 = 0.5;
    pub(super) const SOAP_RMSD: f64 = 0.35;
    pub(super) const SOAP_CUTOFF: f64 = 3.5;
    pub(super) const ESCAPE_EPSILON: f64 = 1.0e-4;
    pub(super) const ESCAPE_AMPLITUDE: f64 = 0.25;
    pub(super) const SCREEN_MARGIN: f64 = 2.0;
    pub(super) const RECORD_GRADIENT: f64 = 1.0e-3;
    /// Screen length of the Lennard-Jones preset.
    pub(super) const SCREEN_STEPS: usize = 25;
    /// Quench length on [`super::Config::for_cluster`].
    ///
    /// The Bayes threshold of [`super::Config::recommended`] is
    /// [`crate::screen::cost_asymmetric_threshold`] at [`SCREEN_STEPS`] and
    /// this length. The closed form `(R - S) / (2 R - S)` is `175/375`,
    /// which reduces to `7/15`. No logged run compares that value with the
    /// quotient at a 1000-step quench, so [`super::Config::recommended`]
    /// keeps this screen pair. [`super::Config::derived`] recomputes the
    /// threshold from the cap.
    pub(super) const RELAX_STEPS: usize = 200;
    /// Quench length on [`super::Config::recommended`].
    ///
    /// The Cambridge global minimisation program sets both iteration caps to
    /// 1000 on its LJ98 deck.
    /// A separate earlier experiment, logged in
    /// `docs/measurements/lj98-quench-cap`, ran LJ98 seeds 0 to 63 on code
    /// that still drew a compacted surface.
    /// That experiment has its own call ceiling, recorded in `traces.tsv`.
    /// At 200 steps, 63088 of 400366 full quenches were unvalidated, and 43
    /// of 64 seeds reached the Leary tetrahedron.
    /// At 1000 steps, 21815 of 353480 full quenches were unvalidated, and 45
    /// of 64 seeds reached it.
    pub(super) const RECOMMENDED_RELAX_STEPS: usize = 1000;
    pub(super) const CONTAINER_RADIUS: f64 = 0.9;
    pub(super) const MIN_SEPARATION: f64 = 0.85;
}

/// Species-aware coefficients multiplied by a derived covalent length scale.
pub(super) struct MolecularPreset;

impl MolecularPreset {
    pub(super) const COVALENT_DIAMETER: f64 = 2.0;
    pub(super) const GROUP_CUTOFF: f64 = 2.5;
    pub(super) const REPACK_RADIUS: f64 = 2.5;
    pub(super) const REPACK_RADIAL_JITTER: f64 = 2.0;
    pub(super) const REPACK_GROUP_SPACING: f64 = 0.15;
}
