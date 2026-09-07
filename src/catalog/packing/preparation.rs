use std::cell::RefCell;
use std::rc::Rc;

use ndarray::{Array2, ArrayView1};

const MAX_RETAINED_BYTES: usize = 8 * 1024 * 1024;

thread_local! {
    static ROWS: RefCell<RowCache> = const { RefCell::new(RowCache::new(MAX_RETAINED_BYTES)) };
}

pub(super) fn packing_rows(coordinates: &[f64]) -> Rc<Array2<f64>> {
    ROWS.with(|cache| {
        cache.borrow_mut().get_or_prepare(coordinates, || {
            crate::soap::local_nu3_z(ArrayView1::from(coordinates), super::PACKING_SPEC, None)
        })
    })
}

struct RowCache {
    max_retained_bytes: usize,
}

impl RowCache {
    const fn new(max_retained_bytes: usize) -> Self {
        Self { max_retained_bytes }
    }

    fn get_or_prepare(
        &mut self,
        coordinates: &[f64],
        prepare: impl FnOnce() -> Array2<f64>,
    ) -> Rc<Array2<f64>> {
        let _ = (self.max_retained_bytes, coordinates);
        Rc::new(prepare())
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{RowCache, packing_rows};
    use ndarray::{Array2, ArrayView1};
    use std::cell::Cell;
    use std::rc::Rc;

    fn prepare_counted(calls: &Cell<usize>, shape: (usize, usize)) -> Array2<f64> {
        calls.set(calls.get() + 1);
        Array2::from_elem(shape, calls.get() as f64)
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
}
