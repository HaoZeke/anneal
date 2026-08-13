//! Deterministic diversity-constrained assignment across attraction regions.
//!
//! Region identity is supplied by the fixed-probe transition graph. Candidate
//! utility contains only target-blind search evidence and physical
//! admissibility; known reference or global-minimum morphology is not an input.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Invalid candidate evidence or assignment parameters.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum RegionAssignmentError {
    /// A utility component contains NaN or infinity.
    #[error("region utility contains a nonfinite component")]
    NonFiniteUtility,
    /// At least one admissible source is required for a nonempty assignment.
    #[error("region assignment has no admissible source")]
    NoAdmissibleSource,
    /// A positive family cap is required.
    #[error("region assignment family cap must be positive")]
    ZeroFamilyCap,
    /// The requested slots exceed the capacity allowed by the family cap.
    #[error("region assignment family cap cannot fill every requested slot")]
    InsufficientCapacity,
}

/// Target-blind terms used to rank a region frontier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionUtility {
    /// Posterior uncertainty in fixed-probe outgoing dynamics.
    pub transition_uncertainty: f64,
    /// Inverse current chain occupancy of the attraction region.
    pub inverse_occupancy: f64,
    /// Evidence that an observed edge leaves the current attraction region.
    pub outgoing_frontier: f64,
    /// Compatibility with geometry, frozen atoms, cell, and molecular mode.
    pub geometry_compatibility: f64,
    /// Charged-work or transport cost needed to access the frontier.
    pub access_cost: f64,
}

impl RegionUtility {
    fn validate(self) -> Result<(), RegionAssignmentError> {
        if [
            self.transition_uncertainty,
            self.inverse_occupancy,
            self.outgoing_frontier,
            self.geometry_compatibility,
            self.access_cost,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            Ok(())
        } else {
            Err(RegionAssignmentError::NonFiniteUtility)
        }
    }

    /// Additive target-blind utility used for deterministic ordering.
    pub fn score(self) -> f64 {
        self.transition_uncertainty
            + self.inverse_occupancy
            + self.outgoing_frontier
            + self.geometry_compatibility
            - self.access_cost
    }
}

/// One source chain or observed frontier eligible for population assignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionCandidate {
    source: u32,
    region: usize,
    utility: RegionUtility,
    admissible: bool,
}

impl RegionCandidate {
    /// Construct an admissible candidate.
    pub fn new(
        source: u32,
        region: usize,
        utility: RegionUtility,
    ) -> Result<Self, RegionAssignmentError> {
        utility.validate()?;
        Ok(Self {
            source,
            region,
            utility,
            admissible: true,
        })
    }

    /// Mark whether the candidate satisfies physical constraints.
    pub fn with_admissible(mut self, admissible: bool) -> Self {
        self.admissible = admissible;
        self
    }

    /// Source replica identifier.
    pub fn source(self) -> u32 {
        self.source
    }

    /// Fixed-probe attraction-region identifier.
    pub fn region(self) -> usize {
        self.region
    }

    /// Whether geometry and system constraints permit adoption.
    pub fn is_admissible(self) -> bool {
        self.admissible
    }

    /// Scalar target-blind assignment utility.
    pub fn score(self) -> f64 {
        self.utility.score()
    }
}

/// Assign a fixed population with attraction-region coverage before duplicates.
pub fn diversity_constrained_assignment(
    candidates: &[RegionCandidate],
    slots: usize,
    max_family_size: usize,
) -> Result<Vec<u32>, RegionAssignmentError> {
    if slots == 0 {
        return Ok(Vec::new());
    }
    if max_family_size == 0 {
        return Err(RegionAssignmentError::ZeroFamilyCap);
    }
    let mut ordered = candidates
        .iter()
        .copied()
        .filter(RegionCandidate::is_admissible)
        .collect::<Vec<_>>();
    if ordered.is_empty() {
        return Err(RegionAssignmentError::NoAdmissibleSource);
    }
    ordered.sort_by(candidate_order);
    let unique_sources = ordered
        .iter()
        .map(|candidate| candidate.source)
        .collect::<BTreeSet<_>>();
    if unique_sources.len().saturating_mul(max_family_size) < slots {
        return Err(RegionAssignmentError::InsufficientCapacity);
    }

    let mut selected = Vec::with_capacity(slots);
    let mut covered_regions = BTreeSet::new();
    let mut family_sizes = BTreeMap::<u32, usize>::new();
    for candidate in &ordered {
        if selected.len() == slots {
            break;
        }
        if covered_regions.insert(candidate.region) {
            selected.push(candidate.source);
            *family_sizes.entry(candidate.source).or_default() += 1;
        }
    }
    while selected.len() < slots {
        let candidate = ordered
            .iter()
            .find(|candidate| {
                family_sizes.get(&candidate.source).copied().unwrap_or(0) < max_family_size
            })
            .expect("capacity check guarantees an available source");
        selected.push(candidate.source);
        *family_sizes.entry(candidate.source).or_default() += 1;
    }
    Ok(selected)
}

fn candidate_order(left: &RegionCandidate, right: &RegionCandidate) -> Ordering {
    right
        .score()
        .partial_cmp(&left.score())
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.region.cmp(&right.region))
        .then_with(|| left.source.cmp(&right.source))
}
