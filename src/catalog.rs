//! Search catalogs with explicit event and descriptor-basin identities.

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

pub use archive::{Curiosity, Tessellation, novelty};
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
    CHAMPION_RANK, DEFAULT_MIN_OCCUPIED_FAMILIES, INTERFACE_HORIZON, InterfaceSeat, LeaveFrame,
    LeavePath, OccupancyCertificate, OccupancyLeaveAdopt, PackingRole, assign_interfaces,
    in_interface_ensemble, interface_ladder, is_occupancy_leave_action, leave_shot_accepted,
    leftover_lambda, occupancy_complete, occupancy_complete_at, occupancy_leave_adopt,
    occupancy_min_families, occupancy_retire, occupancy_retire_at, packing_role, promote_one_sided,
    published_energy_score, retis_exchange_adjacent, retis_should_swap, seat_extras,
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
