//! Search catalogs with explicit event and descriptor-basin identities.

pub mod basin;
pub mod calibration;
pub mod census;
pub mod event;
pub mod lj;
pub mod mixing;
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
pub use mixing::{
    AttractorStrength, MIXED_RHAT, MixingEvidence, certified_global_minimum, explore_collapsed,
    invert_mixing, mixed, rhat_series, stronger,
};
pub use packing::{
    PACKING_MERGE, PackingBook, packing_distance, packing_fingerprint, packing_vector, same_packing,
};
pub use signature::{DescriptorSignature, EngineSignature, SignatureDigest, SystemSignature};
pub use validator::{
    CandidateRecord, CandidateValidator, FreshEvaluation, GradientSource, NumericField,
    QuenchStatus, ValidatedCandidate, ValidationFailure, ValidatorConfig, euclidean_gradient_norm,
};

/// Compatibility name for the local-topology event catalog.
#[deprecated(since = "0.9.0", note = "use EventCatalog")]
pub type Catalog = EventCatalog;
