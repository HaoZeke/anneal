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
pub mod leave_learn;
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
    PRODUCTION_MINIMUM_VISITS, PRODUCTION_SINGLETON_BUDGET, PRODUCTION_UNSEEN_MASS_DEN,
    PRODUCTION_UNSEEN_MASS_NUM, gt_min_visits,
};
pub use event::{Event, EventCatalog, TopologyRecord};
pub use hyperband::{
    DEFAULT_MAX_RESOURCE, EnsembleVerdict, MIN_RESOURCE, REDUCTION_FACTOR, WalkRecord,
    current_rung, keep_ids, prune, rungs, verdict,
};
pub use leave_learn::{
    ACTION_EXPLORE, ACTION_LEAVE, ACTION_LOCAL, LEAVE_EI_PROBES, LeaveLearner, cover_arm_count,
    credit_action, fivefold_arm, leave_best, leave_ei_open, observe_leave, pick_leave_action,
    pick_leave_cover, pick_leave_cover_ei,
};
pub use mixing::{
    AttractorStrength, CERTIFY_CHAINS, CERTIFY_DRAWS_PER_HALF, CERTIFY_MIN_SAMPLES,
    CERTIFY_SPLIT_HALVES, MIXED_RHAT, MixingEvidence, certified_global_minimum, explore_collapsed,
    explore_must_leave, invert_mixing, mixed, occupant_rhat, rhat_series, rhat_split, stronger,
};
pub use occupancy::{
    CHAMPION_RANK, COMPACT_RMAX_OVER_CBRT, DEFAULT_MIN_OCCUPIED_FAMILIES, ESTY_Z95,
    INTERFACE_HORIZON, InterfaceSeat, LEAVE_REFUSAL_DWELL, LEFTOVER_SAT_DWELL, LeaveFrame,
    LeavePath, OCCUPANCY_EI_MIN_OBS, OCCUPANCY_SEAM_CONDUCTANCE, OccupancyBookMap,
    OccupancyCertificate, OccupancyCompact, OccupancyFes, OccupancyFesError, OccupancyFold,
    OccupancyLandfoldPoint, OccupancyLeaveAdopt, OccupancyLeaveTarget, PackingRole,
    assign_interfaces, hops_per_core_hour, in_interface_ensemble, interface_ladder,
    is_occupancy_leave_action, leave_crossing_slices, leave_defers, leave_shot_accepted,
    leftover_birth_probability, leftover_dwell_from_census, leftover_esty_stable,
    leftover_esty_upper, leftover_esty_var, leftover_hatch_stable, leftover_lambda,
    leftover_sat_dwell, lens_ring_displacement, occupancy_book_holes, occupancy_compact,
    occupancy_complete, occupancy_complete_at, occupancy_ei_exhausted, occupancy_family_floor,
    occupancy_fes, occupancy_fes_delta, occupancy_fes_from_histograms, occupancy_is_cluster,
    occupancy_landfold_floor, occupancy_landfold_split, occupancy_leave_adopt,
    occupancy_leave_by_birth, occupancy_leave_by_ei, occupancy_leave_new_class,
    occupancy_leave_new_packing, occupancy_leave_target, occupancy_map_floor, occupancy_map_fold,
    occupancy_map_from_histograms, occupancy_map_split, occupancy_min_families, occupancy_retire,
    occupancy_retire_at, occupancy_ring_census, occupancy_ring_class_changed, occupancy_ring_floor,
    occupancy_ring_profile, occupancy_ring_split, occupancy_sparsify_book,
    occupancy_sparsify_packing, packing_role, pitman_yor_p_new, promote_one_sided,
    published_energy_score, retis_exchange_adjacent, retis_should_swap, ring_leave_weight,
    ring_novelty, seat_extras,
};
pub use packing::{
    ENVIRONMENT_RADIUS, GoodTuringSample, PACKING_LINK, PACKING_MERGE, PACKING_MOVE_EPS,
    PACKING_REFERENCE_DRAWS, PACKING_SPEC, PackingBook, PackingPave, PackingReference,
    SEAM_BIN_WIDTH, SEAM_BINS, SEAM_BURST_SHOTS, SEAM_DOORWAY_GAP, SEAM_DOORWAY_WINDOW,
    SEAM_WINDOW, SeamBank, credit_packing_deposit, deliver_frontier_post, different_decaf_family,
    different_packing_family, drain_frontier_arrivals, leaves_packing, leftover_arrivals_saturated,
    include_packing_reference, offer_frontier_post, offer_known_minimum, packing_communities,
    packing_community_count, packing_distance, packing_fingerprint, packing_link_labels,
    packing_reference_book, packing_references, packing_seam_gap, packing_vector,
    remember_packing_reference, same_packing, set_packing_references, take_frontier_posts,
    take_known_minima,
};
pub use signature::{DescriptorSignature, EngineSignature, SignatureDigest, SystemSignature};
pub use validator::{
    CandidateRecord, CandidateValidator, FreshEvaluation, GradientSource, NumericField,
    QuenchStatus, ValidatedCandidate, ValidationFailure, ValidatorConfig, euclidean_gradient_norm,
};

/// Compatibility name for the local-topology event catalog.
#[deprecated(since = "0.9.0", note = "use EventCatalog")]
pub type Catalog = EventCatalog;
