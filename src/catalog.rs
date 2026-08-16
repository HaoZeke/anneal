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

pub use basin::{
    ActiveBasinEntry, AdmissionOutcome, AdmissionRejection, BasinCatalog, BasinCatalogError,
};
pub use calibration::{
    CalibrationError, CalibrationPair, CensusCalibration, EmpiricalQuantileMethod,
    MINIMUM_CENSUS_CALIBRATION_PAIRS, calibrate_census_radius,
};
pub use census::{BasinCensus, BasinId, CensusEntry, CensusError, CensusObservation};
pub use event::{Event, EventCatalog, TopologyRecord};
pub use hyperband::{
    DEFAULT_MAX_RESOURCE, EnsembleVerdict, MIN_RESOURCE, REDUCTION_FACTOR, WalkRecord,
    current_rung, keep_ids, prune, rungs, verdict,
};
pub use mixing::{
    AttractorStrength, MIXED_RHAT, MixingEvidence, certified_global_minimum, explore_collapsed,
    explore_must_leave, invert_mixing, mixed, rhat_series, stronger,
};
pub use occupancy::{
    OccupancyCertificate, is_occupancy_leave_action, occupancy_complete, occupancy_retire,
    published_energy_score,
};
pub use packing::{
    PACKING_MERGE, PACKING_MOVE_EPS, PackingBook, packing_distance, packing_fingerprint,
    packing_vector, same_packing,
};
pub use signature::{DescriptorSignature, EngineSignature, SignatureDigest, SystemSignature};
pub use validator::{
    CandidateRecord, CandidateValidator, FreshEvaluation, GradientSource, NumericField,
    QuenchStatus, ValidatedCandidate, ValidationFailure, ValidatorConfig, euclidean_gradient_norm,
};

/// Compatibility name for the local-topology event catalog.
#[deprecated(since = "0.9.0", note = "use EventCatalog")]
pub type Catalog = EventCatalog;
