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
        let prepared = Arc::new(
            SortedPairs {
                n_points: coordinates.len() / 3,
            }
            .prepare(coordinates)?,
        );
        let bytes = key.len() * std::mem::size_of::<u64>() + prepared.payload_bytes();
        // Oversized spectra bypass storage without evicting reusable entries.
        if bytes <= self.max_payload_bytes {
            while self.payload_bytes > self.max_payload_bytes - bytes {
                let oldest = self
                    .order
                    .pop_front()
                    .expect("resident payload has a FIFO entry");
                let removed = self
                    .entries
                    .remove(&oldest)
                    .expect("FIFO keys are resident");
                self.payload_bytes -=
                    oldest.len() * std::mem::size_of::<u64>() + removed.payload_bytes();
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
        let cache = self
            .cache
            .lock()
            .expect("pair-spectrum cache lock poisoned");
        PairCacheStats {
            hits: cache.hits,
            preparations: cache.preparations,
            entries: cache.entries.len(),
            payload_bytes: cache.payload_bytes,
        }
    }

    fn lower_bound(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> Option<f64> {
        let (left, right) = {
            let mut cache = self
                .cache
                .lock()
                .expect("pair-spectrum cache lock poisoned");
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
        if self
            .lower_bound(left, right)
            .is_some_and(|lower| lower > self.witness.radius)
        {
            ExactStructureRelation::Distinct
        } else {
            self.witness.relation(left, right)
        }
    }

    fn equivalent_structures(&self, left: StructureView<'_>, right: StructureView<'_>) -> bool {
        self.relation_structures(left, right).is_equivalent()
    }

    fn relation_structures(
        &self,
        left: StructureView<'_>,
        right: StructureView<'_>,
    ) -> ExactStructureRelation {
        if left.context != right.context
            || self
                .lower_bound(left.coordinates, right.coordinates)
                .is_some_and(|lower| lower > self.witness.radius)
        {
            ExactStructureRelation::Distinct
        } else {
            self.witness.relation_structures(left, right)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::pair_spectrum_preparation_count;
    use crate::pes_exploration::StructureContext;
    use ndarray::{Array1, array};

    fn witness() -> IraStructureWitness {
        IraStructureWitness { kmax_factor: 1.8, radius: 0.1 }
    }

    #[test]
    fn warm_equivalent_survivors_do_not_prepare_spectra_in_native_fallback() {
        let coordinates = array![0.0, 0.0, 0.0, 1.3, 0.1, 0.0, 0.2, 1.7, 0.3, 0.1, 0.3, 2.1];
        let cached = witness().with_pair_cache(8192);
        assert!(cached.equivalent(coordinates.view(), coordinates.view()));
        let preparations = pair_spectrum_preparation_count();
        assert!(cached.equivalent(coordinates.view(), coordinates.view()));
        assert_eq!(pair_spectrum_preparation_count(), preparations);
    }

    #[test]
    fn warm_contextual_survivors_do_not_prepare_spectra_in_native_fallback() {
        let coordinates = array![0.0, 0.0, 0.0, 1.3, 0.1, 0.0, 0.2, 1.7, 0.3, 0.1, 0.3, 2.1];
        for species in [None, Some(vec![1; 4])] {
            let context = StructureContext::new(species, None, Some("cache-cost".into()));
            let structure = StructureView { coordinates: coordinates.view(), context: &context };
            let cached = witness().with_pair_cache(8192);
            assert!(cached.equivalent_structures(structure, structure));
            let preparations = pair_spectrum_preparation_count();
            assert!(cached.equivalent_structures(structure, structure));
            assert_eq!(pair_spectrum_preparation_count(), preparations);
        }
    }

    #[test]
    fn warm_homometric_survivors_retain_native_rejection_without_preparation() {
        let line = |points: &[f64]| Array1::from_iter(points.iter().flat_map(|&x| [x, 0.0, 0.0]));
        let left = line(&[0.0, 1.0, 4.0, 10.0, 12.0, 17.0]);
        let right = line(&[0.0, 1.0, 8.0, 11.0, 13.0, 17.0]);
        assert_eq!(SortedPairs { n_points: 6 }.bottleneck_lower_bound(left.view(), right.view()), Some(0.0));
        let cached = witness().with_pair_cache(8192);
        assert!(!cached.equivalent(left.view(), right.view()));
        let preparations = pair_spectrum_preparation_count();
        assert!(!cached.equivalent(left.view(), right.view()));
        assert_eq!(pair_spectrum_preparation_count(), preparations);
    }
}
