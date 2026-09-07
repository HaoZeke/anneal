//! Exact descriptor rows for the fixed packing specification and no species.
//!
//! Each thread retains at most 8 MiB of complete coordinate keys and row scalar
//! payloads. The bound excludes queue and reference-count metadata, and callers
//! can keep evicted rows alive through their own `Rc` handles. Codebooks and
//! histograms remain properties of each individual packing book.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use ndarray::{Array2, ArrayView1};

const MAX_RETAINED_BYTES: usize = 8 * 1024 * 1024;

thread_local! {
    static ROWS: RefCell<RowCache> = const { RefCell::new(RowCache::new(MAX_RETAINED_BYTES)) };
    #[cfg(test)]
    static PREPARATIONS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn packing_rows(coordinates: &[f64]) -> Rc<Array2<f64>> {
    ROWS.with(|cache| {
        cache.borrow_mut().get_or_prepare(coordinates, || {
            #[cfg(test)]
            PREPARATIONS.with(|count| count.set(count.get() + 1));
            crate::soap::local_nu3_z(ArrayView1::from(coordinates), super::PACKING_SPEC, None)
        })
    })
}

struct RowCache {
    max_retained_bytes: usize,
    retained_bytes: usize,
    entries: VecDeque<CachedRows>,
}

struct CachedRows {
    key: Box<[u64]>,
    rows: Rc<Array2<f64>>,
    payload_bytes: usize,
}

impl RowCache {
    const fn new(max_retained_bytes: usize) -> Self {
        Self {
            max_retained_bytes,
            retained_bytes: 0,
            entries: VecDeque::new(),
        }
    }

    fn get_or_prepare(
        &mut self,
        coordinates: &[f64],
        prepare: impl FnOnce() -> Array2<f64>,
    ) -> Rc<Array2<f64>> {
        if coordinates.is_empty()
            || !coordinates.len().is_multiple_of(3)
            || coordinates.iter().any(|value| !value.is_finite())
        {
            return Rc::new(prepare());
        }
        if let Some(entry) = self.entries.iter().find(|entry| {
            entry.key.len() == coordinates.len()
                && entry
                    .key
                    .iter()
                    .zip(coordinates)
                    .all(|(&bits, value)| bits == value.to_bits())
        }) {
            return Rc::clone(&entry.rows);
        }

        let rows = Rc::new(prepare());
        let payload_bytes = coordinates
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|key_bytes| {
                rows.len()
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|row_bytes| key_bytes.checked_add(row_bytes))
            });
        let Some(payload_bytes) = payload_bytes.filter(|&bytes| bytes <= self.max_retained_bytes)
        else {
            return rows;
        };

        let key = coordinates.iter().map(|value| value.to_bits()).collect();
        while self.retained_bytes > self.max_retained_bytes - payload_bytes {
            let oldest = self
                .entries
                .pop_front()
                .expect("retained payload has an entry");
            self.retained_bytes -= oldest.payload_bytes;
        }
        self.retained_bytes += payload_bytes;
        self.entries.push_back(CachedRows {
            key,
            rows: Rc::clone(&rows),
            payload_bytes,
        });
        rows
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PACKING_MOVE_EPS, PackingBook, nearby_packing};
    use super::{MAX_RETAINED_BYTES, PREPARATIONS, ROWS, RowCache, packing_rows};
    use ndarray::{Array2, ArrayView1};
    use std::cell::Cell;
    use std::rc::Rc;

    fn prepare_counted(calls: &Cell<usize>, shape: (usize, usize)) -> Array2<f64> {
        calls.set(calls.get() + 1);
        Array2::from_elem(shape, calls.get() as f64)
    }

    struct ScopedRows {
        previous: Option<RowCache>,
        previous_preparations: usize,
    }

    impl ScopedRows {
        fn new(capacity: usize) -> Self {
            Self {
                previous: Some(ROWS.with(|cache| cache.replace(RowCache::new(capacity)))),
                previous_preparations: PREPARATIONS.with(|count| count.replace(0)),
            }
        }
    }

    impl Drop for ScopedRows {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                ROWS.with(|cache| cache.replace(previous));
            }
            PREPARATIONS.with(|count| count.set(self.previous_preparations));
        }
    }

    fn preparation_count() -> usize {
        PREPARATIONS.with(Cell::get)
    }

    fn load_xyz(text: &str) -> Vec<f64> {
        text.lines()
            .skip(2)
            .filter(|line| !line.trim().is_empty())
            .flat_map(|line| line.split_whitespace().skip(1).take(3))
            .map(|coordinate| coordinate.parse().unwrap())
            .collect()
    }

    fn fixture_pairs() -> [(Vec<f64>, Vec<f64>); 2] {
        [
            (
                load_xyz(include_str!("../../../tests/fixtures/lj38_ico.xyz")),
                load_xyz(include_str!("../../../tests/fixtures/lj38_fcc.xyz")),
            ),
            (
                load_xyz(include_str!("../../../tests/fixtures/lj75_ico.xyz")),
                load_xyz(include_str!("../../../tests/fixtures/lj75_marks.xyz")),
            ),
        ]
    }

    fn translated(coordinates: &[f64]) -> Vec<f64> {
        coordinates
            .iter()
            .enumerate()
            .map(|(index, value)| value + [0.31, -0.2, 0.17][index % 3])
            .collect()
    }

    fn permuted(coordinates: &[f64]) -> Vec<f64> {
        coordinates
            .chunks_exact(3)
            .rev()
            .flatten()
            .copied()
            .collect()
    }

    fn perturbed(coordinates: &[f64]) -> Vec<f64> {
        let mut changed = coordinates.to_vec();
        changed[0] += 0.08;
        changed[4] -= 0.07;
        changed
    }

    #[derive(Debug, PartialEq)]
    struct PairState {
        leaders: Vec<Vec<f64>>,
        families: Vec<Vec<f64>>,
        visits: Vec<u64>,
        well_visits: Vec<u64>,
        version: u64,
        community_parent: Vec<usize>,
        remembered: Vec<(Vec<f64>, Vec<f64>, bool)>,
        histograms: [Vec<f64>; 2],
        assigned_histograms: [Vec<f64>; 2],
        nearby: bool,
    }

    fn pair_state(here: &[f64], other: &[f64]) -> PairState {
        let mut book = PackingBook::default();
        book.observe(here).unwrap();
        book.observe(other).unwrap();
        let histograms = [
            book.histogram(here).unwrap(),
            book.histogram(other).unwrap(),
        ];
        let assigned_histograms = [
            book.assign_histogram(here).unwrap(),
            book.assign_histogram(other).unwrap(),
        ];
        PairState {
            leaders: book.env_leaders.clone(),
            families: book.families.clone(),
            visits: book.visits.clone(),
            well_visits: book.well_visits.clone(),
            version: book.version,
            community_parent: book.community_parent.clone(),
            remembered: book
                .histogram_cache
                .borrow()
                .iter()
                .map(|entry| {
                    (
                        entry.coordinates.clone(),
                        entry.histogram.clone(),
                        entry.grown,
                    )
                })
                .collect(),
            histograms,
            assigned_histograms,
            nearby: nearby_packing(here, other),
        }
    }

    fn pair_scenarios(capacity: usize) -> Vec<PairState> {
        let _scope = ScopedRows::new(capacity);
        let mut states = Vec::new();
        for (here, other) in fixture_pairs() {
            let variants = [
                (here.clone(), other.clone()),
                (translated(&here), translated(&other)),
                (permuted(&here), permuted(&other)),
                (perturbed(&here), perturbed(&other)),
            ];
            let mut third = here.clone();
            third[0] += 0.12;
            for (left, right) in variants {
                for (a, b) in [(&left[..], &right[..]), (&right[..], &left[..])] {
                    let first = pair_state(a, b);
                    states.push(pair_state(a, &third));
                    states.push(pair_state(&third, b));
                    let repeated = pair_state(a, b);
                    assert_eq!(first, repeated, "a third peer cannot alter a pair book");
                    states.push(first);
                    states.push(repeated);
                }
            }
        }
        states
    }

    #[test]
    fn identical_coordinate_content_prepares_once_and_reuses_the_rows() {
        let mut cache = RowCache::new(128);
        let calls = Cell::new(0);
        let coordinates = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
        let first = cache.get_or_prepare(&coordinates, || prepare_counted(&calls, (2, 3)));
        let distinct_allocation = coordinates.to_vec();
        let second = cache.get_or_prepare(&distinct_allocation, || prepare_counted(&calls, (2, 3)));

        assert_eq!(
            calls.get(),
            1,
            "equal coordinate content has one preparation"
        );
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(first.as_ref(), &Array2::from_elem((2, 3), 1.0));
    }

    #[test]
    fn every_ordered_coordinate_bit_participates_in_the_key() {
        let mut cache = RowCache::new(1024);
        let calls = Cell::new(0);
        let tiny_drift = f64::from_bits(1.0_f64.to_bits() + 1);
        let inputs = [
            [1.0, 0.0, 0.0],
            [tiny_drift, 0.0, 0.0],
            [1.0, -0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut prepared = Vec::new();
        for (index, coordinates) in inputs.iter().enumerate() {
            let rows = cache.get_or_prepare(coordinates, || prepare_counted(&calls, (1, 1)));
            assert_eq!(rows[[0, 0]], (index + 1) as f64);
            prepared.push(rows);
        }
        assert_eq!(calls.get(), inputs.len());
        for (coordinates, expected) in inputs.iter().zip(&prepared) {
            let repeated = cache.get_or_prepare(coordinates, || prepare_counted(&calls, (1, 1)));
            assert!(Rc::ptr_eq(&repeated, expected));
        }
        assert_eq!(calls.get(), inputs.len());
    }

    #[test]
    fn retained_key_and_row_bytes_are_bounded_with_fifo_eviction() {
        // Three coordinate keys and four row elements occupy 56 payload bytes.
        let entry_bytes = 3 * std::mem::size_of::<u64>() + 4 * std::mem::size_of::<f64>();
        let mut cache = RowCache::new(2 * entry_bytes);
        let calls = Cell::new(0);
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [3.0, 0.0, 0.0];
        let first_a = cache.get_or_prepare(&a, || prepare_counted(&calls, (2, 2)));
        assert_eq!(cache.retained_bytes(), entry_bytes);
        let first_b = cache.get_or_prepare(&b, || prepare_counted(&calls, (2, 2)));
        assert_eq!(cache.retained_bytes(), 2 * entry_bytes);
        let hit_a = cache.get_or_prepare(&a, || prepare_counted(&calls, (2, 2)));
        assert!(Rc::ptr_eq(&hit_a, &first_a));
        assert_eq!(calls.get(), 2);

        cache.get_or_prepare(&c, || prepare_counted(&calls, (2, 2)));
        assert_eq!(cache.retained_bytes(), 2 * entry_bytes);
        let hit_b = cache.get_or_prepare(&b, || prepare_counted(&calls, (2, 2)));
        assert!(
            Rc::ptr_eq(&hit_b, &first_b),
            "a cache hit must not reorder FIFO entries"
        );
        assert_eq!(calls.get(), 3);
        let replacement_a = cache.get_or_prepare(&a, || prepare_counted(&calls, (2, 2)));
        assert!(!Rc::ptr_eq(&replacement_a, &first_a));
        assert_eq!(calls.get(), 4);
        assert_eq!(cache.retained_bytes(), 2 * entry_bytes);
    }

    #[test]
    fn oversized_rows_are_not_retained_and_do_not_evict_existing_entries() {
        let entry_bytes = 3 * std::mem::size_of::<u64>() + 4 * std::mem::size_of::<f64>();
        let mut cache = RowCache::new(2 * entry_bytes);
        let calls = Cell::new(0);
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let oversized = [3.0, 0.0, 0.0];
        let first_a = cache.get_or_prepare(&a, || prepare_counted(&calls, (2, 2)));
        let first_b = cache.get_or_prepare(&b, || prepare_counted(&calls, (2, 2)));
        assert_eq!(cache.retained_bytes(), 2 * entry_bytes);

        let large = cache.get_or_prepare(&oversized, || prepare_counted(&calls, (4, 4)));
        let repeated_large = cache.get_or_prepare(&oversized, || prepare_counted(&calls, (4, 4)));
        assert!(!Rc::ptr_eq(&large, &repeated_large));
        assert_eq!(
            calls.get(),
            4,
            "oversized rows require fresh preparation each time"
        );
        assert_eq!(cache.retained_bytes(), 2 * entry_bytes);
        let hit_a = cache.get_or_prepare(&a, || prepare_counted(&calls, (2, 2)));
        let hit_b = cache.get_or_prepare(&b, || prepare_counted(&calls, (2, 2)));
        assert!(Rc::ptr_eq(&hit_a, &first_a));
        assert!(Rc::ptr_eq(&hit_b, &first_b));
        assert_eq!(
            calls.get(),
            4,
            "oversized entries preserve both retained entries"
        );
        assert_eq!(cache.retained_bytes(), 2 * entry_bytes);
    }

    #[test]
    fn packing_rows_preserve_the_fixed_spec_species_free_calculation() {
        let coordinates = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
        let expected = crate::soap::local_nu3_z(
            ArrayView1::from(&coordinates[..]),
            super::super::PACKING_SPEC,
            None,
        );
        let actual = packing_rows(&coordinates);
        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn invalid_coordinates_compute_without_retention_or_eviction() {
        let mut cache = RowCache::new(128);
        let calls = Cell::new(0);
        let valid = [1.0, 0.0, 0.0];
        let retained = cache.get_or_prepare(&valid, || prepare_counted(&calls, (1, 1)));
        let retained_bytes = cache.retained_bytes();
        let invalid = [
            vec![],
            vec![1.0, 0.0],
            vec![f64::NAN, 0.0, 0.0],
            vec![f64::INFINITY, 0.0, 0.0],
            vec![f64::NEG_INFINITY, 0.0, 0.0],
        ];
        for coordinates in &invalid {
            let first = cache.get_or_prepare(coordinates, || prepare_counted(&calls, (1, 1)));
            let second = cache.get_or_prepare(coordinates, || prepare_counted(&calls, (1, 1)));
            assert!(!Rc::ptr_eq(&first, &second));
            assert_eq!(cache.retained_bytes(), retained_bytes);
        }
        let hit = cache.get_or_prepare(&valid, || prepare_counted(&calls, (1, 1)));
        assert!(Rc::ptr_eq(&retained, &hit));
        assert_eq!(calls.get(), 1 + 2 * invalid.len());
    }

    #[test]
    fn exact_rows_preserve_pair_books_and_nearby_decisions_across_peer_interleaving() {
        let uncached = pair_scenarios(0);
        let cached = pair_scenarios(MAX_RETAINED_BYTES);
        assert_eq!(cached, uncached);
    }

    #[test]
    fn resident_peer_batches_prepare_each_geometry_once_and_keep_the_pair_local_shortcut() {
        for (here, other) in fixture_pairs() {
            let _scope = ScopedRows::new(MAX_RETAINED_BYTES);
            let peers = [other.clone(), translated(&other), permuted(&other)];
            let cold: Vec<_> = peers
                .iter()
                .map(|peer| nearby_packing(&here, peer))
                .collect();
            assert_eq!(preparation_count(), peers.len() + 1);

            let repeated: Vec<_> = peers
                .iter()
                .map(|peer| nearby_packing(&here, peer))
                .collect();
            assert_eq!(cold, repeated);
            assert_eq!(preparation_count(), peers.len() + 1);

            let mut changed_here = here.clone();
            changed_here[0] += PACKING_MOVE_EPS * 0.25;
            for peer in &peers {
                nearby_packing(&changed_here, peer);
            }
            assert_eq!(
                preparation_count(),
                peers.len() + 2,
                "an exact coordinate change prepares only the changed resident geometry"
            );

            let mut book = PackingBook::default();
            let family = book.observe(&here).unwrap();
            let histogram = book.histogram(&here).unwrap();
            let before_shortcut = preparation_count();
            let mut nearby_here = here.clone();
            nearby_here[0] += PACKING_MOVE_EPS * 0.5;
            assert_eq!(book.observe(&nearby_here), Some(family));
            assert_eq!(book.histogram(&nearby_here), Some(histogram));
            assert_eq!(preparation_count(), before_shortcut);
        }
    }
}
