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
    pub(super) const SYMMETRY_TOLERANCE: f64 = 0.35;
    pub(super) const ESCAPE_EPSILON: f64 = 1.0e-4;
    pub(super) const ESCAPE_AMPLITUDE: f64 = 0.25;
    pub(super) const SCREEN_MARGIN: f64 = 2.0;
    pub(super) const RECORD_GRADIENT: f64 = 1.0e-3;
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
