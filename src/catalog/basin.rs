//! Capacity-limited catalog of validated descriptor-basin representatives.

use super::{BasinId, ValidatedCandidate};

/// Invalid active-catalog configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BasinCatalogError {
    /// At least one active representative must be allowed.
    #[error("active catalog capacity must be positive")]
    ZeroCapacity,
    /// Census radius must be finite and nonnegative.
    #[error("census radius must be finite and nonnegative")]
    InvalidCensusRadius,
    /// Aggregate charged-work budget must be positive.
    #[error("aggregate charged-work budget must be positive")]
    ZeroChargedWorkBudget,
}

/// Reason an otherwise validated candidate did not enter the active catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionRejection {
    /// The active representative for this census basin has lower or equal energy.
    SameBasinNotLower,
    /// At least one geometrically conflicting representative has lower or equal energy.
    ConflictNotLower,
    /// The candidate cannot improve the replaceable entry at capacity.
    CapacityNotLower,
}

/// Deterministic result of one serialized catalog admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionOutcome {
    /// Candidate filled an available active slot.
    Added {
        /// Census basin linked to the new entry.
        basin_id: BasinId,
    },
    /// Candidate improved the representative for its census basin.
    ReplacedSameBasin {
        /// Census basin linked to the replacement.
        basin_id: BasinId,
    },
    /// Candidate replaced every entry in its packing conflict set.
    ReplacedConflicts {
        /// Census basin linked to the replacement.
        basin_id: BasinId,
        /// Evicted census basins in stable active-entry order.
        evicted: Vec<BasinId>,
    },
    /// A conflict-free candidate replaced the worst nonincumbent at capacity.
    ReplacedCapacity {
        /// Census basin linked to the replacement.
        basin_id: BasinId,
        /// Evicted census basin.
        evicted: BasinId,
    },
    /// Candidate left the active catalog unchanged.
    Rejected {
        /// Classified rejection reason.
        reason: AdmissionRejection,
    },
}

/// Active representative retaining validated coordinates and provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveBasinEntry {
    census_id: BasinId,
    census_visits_at_admission: u64,
    validated: ValidatedCandidate,
}

impl ActiveBasinEntry {
    /// Stable identifier in the uncapped census.
    pub fn census_id(&self) -> BasinId {
        self.census_id
    }

    /// Exact census count recorded when this representative entered.
    pub fn census_visits_at_admission(&self) -> u64 {
        self.census_visits_at_admission
    }

    /// Receiving-side validated energy used for catalog comparisons.
    pub fn energy(&self) -> f64 {
        self.validated.fresh.energy
    }

    /// Descriptor carried by the validated candidate.
    pub fn descriptor(&self) -> &[f64] {
        &self.validated.candidate.descriptor
    }

    /// Coordinates carried by the validated candidate.
    pub fn coordinates(&self) -> &[f64] {
        &self.validated.candidate.coordinates
    }

    /// Replica that produced the representative.
    pub fn producer_replica(&self) -> u32 {
        self.validated.candidate.producer_replica
    }

    /// Producer event sequence retained for provenance.
    pub fn event_sequence(&self) -> u64 {
        self.validated.candidate.event_sequence
    }

    /// Full validation evidence retained by the catalog.
    pub fn validated(&self) -> &ValidatedCandidate {
        &self.validated
    }
}

/// Finite active catalog with separated representatives and a monotone incumbent.
#[derive(Debug, Clone)]
pub struct BasinCatalog {
    capacity: usize,
    census_radius: f64,
    total_charged_work: u64,
    entries: Vec<ActiveBasinEntry>,
    raw_initial_scale: Option<f64>,
    initial_threshold: Option<f64>,
    packing_threshold: Option<f64>,
    largest_charged_work: u64,
    version: u64,
}

impl BasinCatalog {
    /// Create an empty active catalog and its aggregate-work threshold schedule.
    pub fn new(
        capacity: usize,
        census_radius: f64,
        total_charged_work: u64,
    ) -> Result<Self, BasinCatalogError> {
        if capacity == 0 {
            return Err(BasinCatalogError::ZeroCapacity);
        }
        if !census_radius.is_finite() || census_radius < 0.0 {
            return Err(BasinCatalogError::InvalidCensusRadius);
        }
        if total_charged_work == 0 {
            return Err(BasinCatalogError::ZeroChargedWorkBudget);
        }
        Ok(Self {
            capacity,
            census_radius,
            total_charged_work,
            entries: Vec::new(),
            raw_initial_scale: None,
            initial_threshold: None,
            packing_threshold: None,
            largest_charged_work: 0,
            version: 0,
        })
    }

    /// Compatibility constructor for an empty catalog without a work schedule.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            census_radius: 0.0,
            total_charged_work: 1,
            entries: Vec::new(),
            raw_initial_scale: None,
            initial_threshold: None,
            packing_threshold: None,
            largest_charged_work: 0,
            version: 0,
        }
    }

    /// Apply one serialized admission decision.
    pub fn admit(
        &mut self,
        basin_id: BasinId,
        census_visits: u64,
        validated: ValidatedCandidate,
    ) -> AdmissionOutcome {
        let threshold = self.packing_threshold.unwrap_or(self.census_radius);
        let same_index = self
            .entries
            .iter()
            .position(|entry| entry.census_id == basin_id);
        let conflicts = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                (entry.census_id == basin_id
                    || descriptor_distance(entry.descriptor(), &validated.candidate.descriptor)
                        < threshold)
                    .then_some(index)
            })
            .collect::<Vec<_>>();

        if let Some(index) = same_index
            && validated.fresh.energy >= self.entries[index].energy()
        {
            return AdmissionOutcome::Rejected {
                reason: AdmissionRejection::SameBasinNotLower,
            };
        }
        if !conflicts.is_empty() {
            if conflicts
                .iter()
                .any(|&index| validated.fresh.energy >= self.entries[index].energy())
            {
                return AdmissionOutcome::Rejected {
                    reason: AdmissionRejection::ConflictNotLower,
                };
            }
            let evicted = conflicts
                .iter()
                .map(|&index| self.entries[index].census_id)
                .collect::<Vec<_>>();
            let same_only = evicted.len() == 1 && evicted[0] == basin_id;
            self.replace_indices(
                &conflicts,
                ActiveBasinEntry {
                    census_id: basin_id,
                    census_visits_at_admission: census_visits,
                    validated,
                },
            );
            return if same_only {
                AdmissionOutcome::ReplacedSameBasin { basin_id }
            } else {
                AdmissionOutcome::ReplacedConflicts { basin_id, evicted }
            };
        }

        let entry = ActiveBasinEntry {
            census_id: basin_id,
            census_visits_at_admission: census_visits,
            validated,
        };
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
            self.note_mutation();
            self.initialize_threshold_if_full();
            return AdmissionOutcome::Added { basin_id };
        }

        let incumbent_index = self.incumbent_index().expect("a full catalog is nonempty");
        let replacement_index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != incumbent_index)
            .max_by(|(_, left), (_, right)| {
                left.energy()
                    .total_cmp(&right.energy())
                    .then_with(|| left.census_id.cmp(&right.census_id))
            })
            .map(|(index, _)| index)
            .unwrap_or(incumbent_index);
        if entry.energy() >= self.entries[replacement_index].energy() {
            return AdmissionOutcome::Rejected {
                reason: AdmissionRejection::CapacityNotLower,
            };
        }
        let evicted = self.entries[replacement_index].census_id;
        self.entries[replacement_index] = entry;
        self.note_mutation();
        AdmissionOutcome::ReplacedCapacity { basin_id, evicted }
    }

    /// Maximum number of active basin representatives.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of active basin representatives.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the active catalog contains no representatives.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Active entries in deterministic storage order.
    pub fn entries(&self) -> &[ActiveBasinEntry] {
        &self.entries
    }

    /// Look up an active entry through its uncapped census identity.
    pub fn entry(&self, basin_id: BasinId) -> Option<&ActiveBasinEntry> {
        self.entries
            .iter()
            .find(|entry| entry.census_id == basin_id)
    }

    /// Lowest-energy active representative.
    pub fn incumbent(&self) -> Option<&ActiveBasinEntry> {
        self.incumbent_index().map(|index| &self.entries[index])
    }

    /// Half-mean pair-distance scale measured from the initial population.
    pub fn raw_initial_scale(&self) -> Option<f64> {
        self.raw_initial_scale
    }

    /// Packing-safe initial threshold derived from the initial population.
    pub fn initial_threshold(&self) -> Option<f64> {
        self.initial_threshold
    }

    /// Current nonincreasing packing threshold.
    pub fn packing_threshold(&self) -> Option<f64> {
        self.packing_threshold
    }

    /// Monotone catalog version incremented by active-state mutations.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Lower the packing threshold according to aggregate charged work.
    pub fn update_threshold(&mut self, charged_work: u64) -> Option<f64> {
        let initial = self.initial_threshold?;
        self.largest_charged_work = self.largest_charged_work.max(charged_work);
        let floor = self.census_radius.max(0.4 * initial).min(initial);
        let progress = (self.largest_charged_work as f64 / (0.8 * self.total_charged_work as f64))
            .clamp(0.0, 1.0);
        let scheduled = if initial == 0.0 {
            0.0
        } else {
            initial * (floor / initial).powf(progress)
        };
        let current = self.packing_threshold.unwrap_or(initial);
        let next = current.min(scheduled);
        if next < current {
            self.packing_threshold = Some(next);
            self.note_mutation();
        }
        self.packing_threshold
    }

    /// Removes one entry by census id. Returns whether an entry was removed.
    ///
    /// The diversity-preserving admission on the coordinator uses this to
    /// make room in a full catalog by evicting the highest-energy member of
    /// the most crowded packing family, so a candidate from a family the
    /// catalog does not hold is admitted even when it is not lower than the
    /// catalog's worst entry.
    pub fn evict(&mut self, basin_id: BasinId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.census_id != basin_id);
        let removed = self.entries.len() != before;
        if removed {
            self.note_mutation();
        }
        removed
    }

    fn replace_indices(&mut self, indices: &[usize], replacement: ActiveBasinEntry) {
        let mut remove = indices.iter().copied();
        let mut next = remove.next();
        let mut retained = Vec::with_capacity(self.entries.len() + 1 - indices.len());
        for (index, entry) in self.entries.drain(..).enumerate() {
            if next == Some(index) {
                next = remove.next();
            } else {
                retained.push(entry);
            }
        }
        retained.push(replacement);
        self.entries = retained;
        self.note_mutation();
    }

    fn initialize_threshold_if_full(&mut self) {
        if self.initial_threshold.is_some() || self.entries.len() != self.capacity {
            return;
        }
        if self.entries.len() < 2 {
            self.raw_initial_scale = Some(self.census_radius);
            self.initial_threshold = Some(self.census_radius);
            self.packing_threshold = Some(self.census_radius);
            return;
        }
        let mut distance_sum = 0.0;
        let mut minimum_distance = f64::INFINITY;
        let mut pair_count = 0u64;
        for left in 0..self.entries.len() {
            for right in left + 1..self.entries.len() {
                let distance = descriptor_distance(
                    self.entries[left].descriptor(),
                    self.entries[right].descriptor(),
                );
                distance_sum += distance;
                minimum_distance = minimum_distance.min(distance);
                pair_count += 1;
            }
        }
        let raw_scale = 0.5 * distance_sum / pair_count as f64;
        let safe_threshold = raw_scale.min(minimum_distance);
        self.raw_initial_scale = Some(raw_scale);
        self.initial_threshold = Some(safe_threshold);
        self.packing_threshold = Some(safe_threshold);
    }

    fn incumbent_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                left.energy()
                    .total_cmp(&right.energy())
                    .then_with(|| left.census_id.cmp(&right.census_id))
            })
            .map(|(index, _)| index)
    }

    fn note_mutation(&mut self) {
        self.version = self.version.saturating_add(1);
    }
}

fn descriptor_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}
