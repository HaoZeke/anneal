//! Search catalogs with explicit event and descriptor-basin identities.

/// Quality-diversity archive: tessellation, curiosity, novelty.
pub mod archive;

/// Cells the occupancy archive tessellates descriptor space into.
///
/// A parameter rather than an outcome, which is the whole point:
/// leader-clustered families numbered twenty-two on a twenty-four
/// replica wave, so coverage was a ratio whose denominator grew as it
/// was measured. Set above the wave size so an ensemble cannot occupy
/// every cell and coverage stays informative, and below the point
/// where cells hold one structure each and the archive is a list.
pub const OCCUPANCY_NICHES: usize = 64;
pub mod basin;
pub mod calibration;
pub mod census;
pub mod event;
pub mod hyperband;
pub mod lj;
pub mod mixing;
pub mod molecular;
pub mod occupancy;
pub mod packing;
pub mod signature;
pub mod validator;

pub use archive::{Archive, Curiosity, novelty};
pub use basin::{
    ActiveBasinEntry, AdmissionOutcome, AdmissionRejection, BasinCatalog, BasinCatalogError,
};
pub use calibration::{
    CalibrationError, CalibrationPair, CensusCalibration, EmpiricalQuantileMethod,
    MINIMUM_CENSUS_CALIBRATION_PAIRS, calibrate_census_radius,
};
pub use census::{
    BasinCensus, BasinId, CensusEntry, CensusError, CensusObservation, PRODUCTION_MAX_UNSEEN_MASS,
    PRODUCTION_MINIMUM_VISITS,
};
pub use event::{Event, EventCatalog, TopologyRecord};
pub use hyperband::{
    DEFAULT_MAX_RESOURCE, EnsembleVerdict, MIN_RESOURCE, REDUCTION_FACTOR, WalkRecord,
    current_rung, keep_ids, prune, rungs, verdict,
};
pub use mixing::{
    AttractorStrength, CERTIFY_MIN_SAMPLES, MIXED_RHAT, MixingEvidence, certified_global_minimum,
    explore_collapsed, explore_must_leave, invert_mixing, mixed, occupant_rhat, rhat_series,
    stronger,
};
pub use occupancy::{
    CHAMPION_RANK, DEFAULT_MIN_OCCUPIED_FAMILIES, INTERFACE_HORIZON, InterfaceSeat,
    LEFTOVER_SAT_DWELL, LeaveFrame, LeavePath, OCCUPANCY_EI_MIN_OBS, OCCUPANCY_SEAM_CONDUCTANCE,
    OccupancyCertificate, OccupancyLeaveAdopt, OccupancyLeaveTarget, PackingRole,
    assign_interfaces, in_interface_ensemble, interface_ladder, is_occupancy_leave_action,
    leave_shot_accepted, leftover_lambda, leftover_sat_dwell, lens_ring_displacement,
    occupancy_complete, occupancy_complete_at, occupancy_ei_exhausted, occupancy_family_floor,
    occupancy_fes_basins, occupancy_fes_from_histograms, occupancy_landfold_floor,
    occupancy_landfold_split, occupancy_leave_adopt, occupancy_leave_target, occupancy_map_floor,
    occupancy_map_from_histograms, occupancy_map_split, occupancy_min_families, occupancy_retire,
    occupancy_retire_at, occupancy_ring_census, occupancy_ring_floor, occupancy_ring_profile,
    occupancy_ring_split, packing_role, promote_one_sided, published_energy_score,
    retis_exchange_adjacent, retis_should_swap, ring_leave_weight, ring_novelty, seat_extras,
};
pub use packing::{
    GoodTuringSample, PACKING_MERGE, PACKING_MOVE_EPS, PackingBook, different_decaf_family,
    leftover_arrivals_saturated, packing_distance, packing_fingerprint, packing_vector,
    same_packing,
};
pub use signature::{DescriptorSignature, EngineSignature, SignatureDigest, SystemSignature};
pub use validator::{
    CandidateRecord, CandidateValidator, FreshEvaluation, GradientSource, NumericField,
    QuenchStatus, ValidatedCandidate, ValidationFailure, ValidatorConfig, euclidean_gradient_norm,
};

/// Compatibility name for the local-topology event catalog.
#[deprecated(since = "0.9.0", note = "use EventCatalog")]
pub type Catalog = EventCatalog;
