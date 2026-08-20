//! Append-only descriptor-basin census with exact visit accounting.

/// Coverage policy: leftover and packing Good--Turing refuse to call
/// the census complete while unseen mass is at least `num/den`.
pub const PRODUCTION_UNSEEN_MASS_NUM: u64 = 1;
/// Denominator of the coverage policy. With the numerator this is
/// \(\alpha = 1/5\).
pub const PRODUCTION_UNSEEN_MASS_DEN: u64 = 5;
/// Largest leftover-singleton count that must stay hatch-stable at
/// the visit floor. `n_min(α, k) = \lfloor (k+1)/α \rfloor`.
pub const PRODUCTION_SINGLETON_BUDGET: u64 = 3;

/// Smallest `n` such that `singleton_budget` leftover singletons stay
/// hatch-stable at ceiling `ceiling_num/ceiling_den`.
///
/// `(k+1)/(n+1) < p/q` iff `n ≥ \lfloor (k+1) q / p \rfloor` when
/// `p` divides `(k+1)q`. For `α = 1/5` and `k = 3` this is 20.
pub const fn gt_min_visits(
    ceiling_num: u64,
    ceiling_den: u64,
    singleton_budget: u64,
) -> u64 {
    (singleton_budget + 1) * ceiling_den / ceiling_num
}

/// Production Good--Turing visit floor: `gt_min_visits(1, 5, 3) = 20`.
pub const PRODUCTION_MINIMUM_VISITS: u64 = gt_min_visits(
    PRODUCTION_UNSEEN_MASS_NUM,
    PRODUCTION_UNSEEN_MASS_DEN,
    PRODUCTION_SINGLETON_BUDGET,
);
/// Production Good--Turing unseen-mass ceiling `α = 1/5`.
pub const PRODUCTION_MAX_UNSEEN_MASS: f64 =
    PRODUCTION_UNSEEN_MASS_NUM as f64 / PRODUCTION_UNSEEN_MASS_DEN as f64;

/// Stable identifier for one immutable census medoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasinId(u64);

impl BasinId {
    /// Construct an identifier from its canonical integer representation.
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Return the canonical integer representation.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// One immutable medoid and its exact visit count.
#[derive(Debug, Clone, PartialEq)]
pub struct CensusEntry {
    id: BasinId,
    medoid: Vec<f64>,
    visits: u64,
}

impl CensusEntry {
    /// Stable basin identifier.
    pub fn id(&self) -> BasinId {
        self.id
    }

    /// Descriptor that opened the basin.
    pub fn medoid(&self) -> &[f64] {
        &self.medoid
    }

    /// Exact number of observations assigned to this basin.
    pub fn visits(&self) -> u64 {
        self.visits
    }
}

/// Result of assigning one validated descriptor observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CensusObservation {
    /// Assigned basin identifier.
    pub basin_id: BasinId,
    /// Whether this observation opened a new basin.
    pub created: bool,
    /// Exact basin count after this observation.
    pub basin_visits: u64,
    /// Exact global count after this observation.
    pub total_visits: u64,
}

/// Configuration or observation error that leaves the census unchanged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CensusError {
    /// Descriptor dimension must be positive.
    #[error("descriptor dimension must be positive")]
    ZeroDescriptorDimension,
    /// Census radius must be finite and nonnegative.
    #[error("census radius must be finite and nonnegative")]
    InvalidRadius,
    /// Observation length differs from the census schema.
    #[error("descriptor dimension is {actual}, expected {expected}")]
    DescriptorDimension {
        /// Dimension fixed by the census.
        expected: usize,
        /// Dimension carried by the observation.
        actual: usize,
    },
    /// An observation contains NaN or infinity.
    #[error("nonfinite descriptor value at index {index}")]
    NonFiniteDescriptor {
        /// Index of the first invalid value.
        index: usize,
    },
    /// A visit or identifier counter cannot be represented by `u64`.
    #[error("census counter overflow")]
    CounterOverflow,
}

/// Exact, uncapped census of fixed-radius descriptor basins.
#[derive(Debug, Clone)]
pub struct BasinCensus {
    descriptor_dim: usize,
    radius_squared: f64,
    entries: Vec<CensusEntry>,
    total_visits: u64,
}

impl BasinCensus {
    /// Create an empty census with a fixed descriptor schema and radius.
    pub fn new(descriptor_dim: usize, radius: f64) -> Result<Self, CensusError> {
        if descriptor_dim == 0 {
            return Err(CensusError::ZeroDescriptorDimension);
        }
        let radius_squared = radius * radius;
        if !radius.is_finite() || radius < 0.0 || !radius_squared.is_finite() {
            return Err(CensusError::InvalidRadius);
        }
        Ok(Self {
            descriptor_dim,
            radius_squared,
            entries: Vec::new(),
            total_visits: 0,
        })
    }

    /// Assign one descriptor to its nearest existing medoid within the radius.
    pub fn observe(&mut self, descriptor: &[f64]) -> Result<CensusObservation, CensusError> {
        self.validate_descriptor(descriptor)?;
        let next_total = self
            .total_visits
            .checked_add(1)
            .ok_or(CensusError::CounterOverflow)?;

        if let Some(index) = self.nearest_within_radius(descriptor) {
            let next_basin_visits = self.entries[index]
                .visits
                .checked_add(1)
                .ok_or(CensusError::CounterOverflow)?;
            self.entries[index].visits = next_basin_visits;
            self.total_visits = next_total;
            return Ok(CensusObservation {
                basin_id: self.entries[index].id,
                created: false,
                basin_visits: next_basin_visits,
                total_visits: next_total,
            });
        }

        let raw_id = u64::try_from(self.entries.len()).map_err(|_| CensusError::CounterOverflow)?;
        let basin_id = BasinId::from_raw(raw_id);
        self.entries.push(CensusEntry {
            id: basin_id,
            medoid: descriptor.to_vec(),
            visits: 1,
        });
        self.total_visits = next_total;
        Ok(CensusObservation {
            basin_id,
            created: true,
            basin_visits: 1,
            total_visits: next_total,
        })
    }

    /// Number of immutable census basins.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the census contains no basins.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries in stable identifier order.
    pub fn entries(&self) -> &[CensusEntry] {
        &self.entries
    }

    /// Look up one entry by stable identifier.
    pub fn entry(&self, id: BasinId) -> Option<&CensusEntry> {
        usize::try_from(id.as_raw())
            .ok()
            .and_then(|index| self.entries.get(index))
            .filter(|entry| entry.id == id)
    }

    /// Exact number of successful observations.
    pub fn total_visits(&self) -> u64 {
        self.total_visits
    }

    /// Classify a descriptor without changing visit counts.
    pub fn basin_for(&self, descriptor: &[f64]) -> Result<Option<BasinId>, CensusError> {
        self.validate_descriptor(descriptor)?;
        Ok(self
            .nearest_within_radius(descriptor)
            .map(|index| self.entries[index].id))
    }

    /// Number of basins observed exactly once.
    pub fn singleton_count(&self) -> u64 {
        u64::try_from(
            self.entries
                .iter()
                .filter(|entry| entry.visits == 1)
                .count(),
        )
        .unwrap_or(u64::MAX)
    }

    /// Good--Turing unseen-mass estimate `N1 / n`, if any visit exists.
    pub fn unseen_mass(&self) -> Option<f64> {
        (self.total_visits != 0).then(|| self.singleton_count() as f64 / self.total_visits as f64)
    }

    /// Whether the production visit floor and unseen-mass threshold are met.
    pub fn is_saturated(&self) -> bool {
        self.saturated_at(PRODUCTION_MINIMUM_VISITS, PRODUCTION_MAX_UNSEEN_MASS)
    }

    /// Evaluate a declared visit floor and strict unseen-mass threshold.
    pub fn saturated_at(&self, minimum_visits: u64, maximum_unseen_mass: f64) -> bool {
        self.total_visits >= minimum_visits
            && self
                .unseen_mass()
                .is_some_and(|mass| mass < maximum_unseen_mass)
    }

    fn validate_descriptor(&self, descriptor: &[f64]) -> Result<(), CensusError> {
        if descriptor.len() != self.descriptor_dim {
            return Err(CensusError::DescriptorDimension {
                expected: self.descriptor_dim,
                actual: descriptor.len(),
            });
        }
        if let Some(index) = descriptor.iter().position(|value| !value.is_finite()) {
            return Err(CensusError::NonFiniteDescriptor { index });
        }
        Ok(())
    }

    fn nearest_within_radius(&self, descriptor: &[f64]) -> Option<usize> {
        let mut nearest = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let distance_squared = descriptor
                .iter()
                .zip(&entry.medoid)
                .map(|(value, medoid)| {
                    let delta = value - medoid;
                    delta * delta
                })
                .sum::<f64>();
            if distance_squared > self.radius_squared {
                continue;
            }
            if nearest.is_none_or(|(_, best_distance)| distance_squared < best_distance) {
                nearest = Some((index, distance_squared));
            }
        }
        nearest.map(|(index, _)| index)
    }
}
