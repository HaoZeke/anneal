//! Capacity-limited catalog of descriptor-defined basin representatives.

/// Active descriptor-basin catalog.
///
/// The explicit name distinguishes descriptor basins from the local-topology
/// events used by k-ART searches.
#[derive(Debug, Clone)]
pub struct BasinCatalog {
    capacity: usize,
}

impl BasinCatalog {
    /// Create an empty catalog with a fixed active-entry capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { capacity }
    }

    /// Maximum number of active basin representatives.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of active basin representatives.
    pub fn len(&self) -> usize {
        0
    }

    /// Whether the active catalog contains no representatives.
    pub fn is_empty(&self) -> bool {
        true
    }
}
