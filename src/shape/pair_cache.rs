//! Per-witness immutable pair spectra; cache membership never certifies identity.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use ndarray::ArrayView1;

use super::IraStructureWitness;
use crate::bias::{PreparedPairSpectrum, SortedPairs};
use crate::pes_exploration::{ExactStructureRelation, ExactStructureWitness, StructureView};

/// Rejection-prefilter cache counters, excluding work in native-match fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairCacheStats {
    /// Coordinate-content lookups served by a stored spectrum.
    pub hits: u64,
    /// Spectrum preparation attempts on cache misses, including invalid inputs.
    pub preparations: u64,
    /// Resident coordinate sets.
    pub entries: usize,
    /// Resident coordinate-key and pair-distance bytes, excluding map metadata.
    pub payload_bytes: usize,
}

struct PairCache {
    entries: HashMap<Arc<[u64]>, Arc<PreparedPairSpectrum>>,
    order: VecDeque<Arc<[u64]>>,
    max_payload_bytes: usize,
    payload_bytes: usize,
    hits: u64,
    preparations: u64,
}

impl PairCache {
    fn prepare(&mut self, coordinates: ArrayView1<f64>) -> Option<Arc<PreparedPairSpectrum>> {
        let key: Vec<u64> = coordinates.iter().map(|value| value.to_bits()).collect();
        if let Some(prepared) = self.entries.get(key.as_slice()) {
            self.hits += 1;
            return Some(Arc::clone(prepared));
        }
        self.preparations += 1;
        let prepared = Arc::new(SortedPairs { n_points: coordinates.len() / 3 }.prepare(coordinates)?);
        let bytes = key.len() * std::mem::size_of::<u64>() + prepared.payload_bytes();
        // Oversized spectra bypass storage without evicting reusable entries.
        if bytes <= self.max_payload_bytes {
            while self.payload_bytes > self.max_payload_bytes - bytes {
                let oldest = self.order.pop_front().expect("resident payload has a FIFO entry");
                let removed = self.entries.remove(&oldest).expect("FIFO keys are resident");
                self.payload_bytes -= oldest.len() * std::mem::size_of::<u64>() + removed.payload_bytes();
            }
            let key: Arc<[u64]> = key.into();
            self.order.push_back(Arc::clone(&key));
            self.entries.insert(key, Arc::clone(&prepared));
            self.payload_bytes += bytes;
        }
        Some(prepared)
    }
}

/// IRA identity witness with a bounded, instance-local pair-spectrum prefilter.
///
/// Complete coordinate bits key immutable spectra; representative replacement
/// cannot reuse stale geometry. FIFO eviction changes cost, not identity.
/// Native survivors use the original witness, including its species and domain
/// checks. This wrapper does not serialize native IRA calls.
pub struct CachedIraStructureWitness {
    witness: IraStructureWitness,
    cache: Mutex<PairCache>,
}

impl IraStructureWitness {
    /// Cache at most `max_payload_bytes` of coordinate keys and sorted distances.
    ///
    /// Map and reference-count metadata are additional to the payload budget.
    /// Zero disables storage. Each wrapper starts cold, so independent search
    /// comparisons need not inherit another arm's prepared geometry.
    pub fn with_pair_cache(self, max_payload_bytes: usize) -> CachedIraStructureWitness {
        CachedIraStructureWitness {
            witness: self,
            cache: Mutex::new(PairCache {
                entries: HashMap::new(),
                order: VecDeque::new(),
                max_payload_bytes,
                payload_bytes: 0,
                hits: 0,
                preparations: 0,
            }),
        }
    }
}

impl CachedIraStructureWitness {
    /// Snapshot of rejection-prefilter work and resident payload.
    pub fn cache_stats(&self) -> PairCacheStats {
        let cache = self.cache.lock().expect("pair-spectrum cache lock poisoned");
        PairCacheStats {
            hits: cache.hits,
            preparations: cache.preparations,
            entries: cache.entries.len(),
            payload_bytes: cache.payload_bytes,
        }
    }

    fn lower_bound(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> Option<f64> {
        let (left, right) = {
            let mut cache = self.cache.lock().expect("pair-spectrum cache lock poisoned");
            (cache.prepare(left)?, cache.prepare(right)?)
        };
        left.bottleneck_lower_bound(&right)
    }
}

impl ExactStructureWitness for CachedIraStructureWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        self.relation(left, right).is_equivalent()
    }

    fn relation(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> ExactStructureRelation {
        if self.lower_bound(left, right).is_some_and(|lower| lower > self.witness.radius) {
            ExactStructureRelation::Distinct
        } else {
            self.witness.relation(left, right)
        }
    }

    fn equivalent_structures(&self, left: StructureView<'_>, right: StructureView<'_>) -> bool {
        self.relation_structures(left, right).is_equivalent()
    }

    fn relation_structures(&self, left: StructureView<'_>, right: StructureView<'_>) -> ExactStructureRelation {
        if left.context != right.context
            || self.lower_bound(left.coordinates, right.coordinates)
                .is_some_and(|lower| lower > self.witness.radius)
        {
            ExactStructureRelation::Distinct
        } else {
            self.witness.relation_structures(left, right)
        }
    }
}
