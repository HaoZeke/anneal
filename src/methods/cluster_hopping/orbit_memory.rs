//! What a chain keeps of its orbit completions, under [`OrbitMemory`].

use ndarray::ArrayView1;

use super::OrbitMemory;

/// Energies within this many energy units of one another, times
/// [`super::Config::energy_scale`], are one energy.
pub(super) const COMPLETED_ENERGY_TOLERANCE: f64 = 1e-6;

/// One kept minimum: its energy, and under [`OrbitMemory::Structure`] its
/// basin descriptor.
#[derive(Debug, Clone)]
struct Kept {
    energy: f64,
    descriptor: Vec<f64>,
}

/// The minima a chain has put through orbit completion, sorted by energy.
#[derive(Debug, Clone)]
pub(super) struct CompletionMemory {
    mode: OrbitMemory,
    tolerance: f64,
    radius: f64,
    kept: Vec<Kept>,
}

fn within(a: &[f64], b: ArrayView1<f64>, radius: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let r2 = radius * radius;
    let mut d2 = 0.0;
    for (p, q) in a.iter().zip(b.iter()) {
        d2 += (p - q) * (p - q);
        if d2 > r2 {
            return false;
        }
    }
    true
}

impl CompletionMemory {
    /// A memory under `mode`, calling two energies one within `tolerance`
    /// and two descriptors one within `radius`.
    pub(super) fn new(mode: OrbitMemory, tolerance: f64, radius: f64) -> Self {
        Self {
            mode,
            tolerance,
            radius,
            kept: Vec::new(),
        }
    }

    /// Whether the memory reads basin descriptors.
    pub(super) fn structural(&self) -> bool {
        self.mode == OrbitMemory::Structure
    }

    /// Whether the minimum at `energy` with descriptor `descriptor` is kept.
    /// The descriptor is read only under [`OrbitMemory::Structure`], where a
    /// minimum without one is never kept.
    pub(super) fn holds(&self, energy: f64, descriptor: Option<ArrayView1<f64>>) -> bool {
        let lo = self
            .kept
            .partition_point(|k| k.energy < energy - self.tolerance);
        let hi = self
            .kept
            .partition_point(|k| k.energy <= energy + self.tolerance);
        let same_energy = &self.kept[lo..hi];
        match (self.mode, descriptor) {
            (OrbitMemory::Off, _) | (OrbitMemory::Structure, None) => false,
            (OrbitMemory::Energy, _) => !same_energy.is_empty(),
            (OrbitMemory::Structure, Some(d)) => same_energy
                .iter()
                .any(|k| within(&k.descriptor, d, self.radius)),
        }
    }

    /// Keeps the minimum at `energy` with descriptor `descriptor`, unless it
    /// is kept already.
    pub(super) fn keep(&mut self, energy: f64, descriptor: Option<ArrayView1<f64>>) {
        if self.mode == OrbitMemory::Off || !energy.is_finite() || self.holds(energy, descriptor) {
            return;
        }
        let descriptor = match (self.mode, descriptor) {
            (OrbitMemory::Structure, Some(d)) => d.to_vec(),
            (OrbitMemory::Structure, None) => return,
            _ => Vec::new(),
        };
        let at = self.kept.partition_point(|k| k.energy < energy);
        self.kept.insert(at, Kept { energy, descriptor });
    }

    /// Returns whether the chain may adopt a completion from the minimum at
    /// `from` to the minimum at `to`. Under [`OrbitMemory::Off`] it keeps
    /// nothing and says yes. Otherwise it keeps `to`, and says yes only when
    /// `to` was not kept before and lies below `from` by more than the tolerance.
    pub(super) fn offer(
        &mut self,
        from: f64,
        to: f64,
        to_descriptor: Option<ArrayView1<f64>>,
    ) -> bool {
        if self.mode == OrbitMemory::Off {
            return true;
        }
        let fresh = !self.holds(to, to_descriptor);
        self.keep(to, to_descriptor);
        fresh && to < from - self.tolerance
    }

    /// Minima kept so far.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.kept.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionMemory, OrbitMemory};
    use ndarray::array;

    #[test]
    fn off_keeps_nothing_and_offers_everything() {
        let mut m = CompletionMemory::new(OrbitMemory::Off, 1e-6, 0.7);
        m.keep(-1.0, None);
        assert_eq!(m.len(), 0);
        assert!(!m.holds(-1.0, None));
        assert!(m.offer(-1.0, -1.0, None));
        assert!(m.offer(-1.0, 0.0, None));
    }

    #[test]
    fn energy_memory_merges_within_the_tolerance() {
        let mut m = CompletionMemory::new(OrbitMemory::Energy, 1e-6, 0.7);
        m.keep(-10.0, None);
        assert!(m.holds(-10.0 + 5e-7, None));
        assert!(!m.holds(-10.0 + 2e-6, None));
        m.keep(-10.0 + 5e-7, None);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn an_offer_is_adopted_only_downhill_onto_a_minimum_not_kept() {
        let mut m = CompletionMemory::new(OrbitMemory::Energy, 1e-6, 0.7);
        assert!(!m.offer(-10.0, -9.0, None), "uphill");
        assert!(!m.offer(-10.0, -10.0, None), "level");
        assert!(m.offer(-10.0, -11.0, None), "downhill and new");
        assert!(!m.offer(-10.0, -11.0, None), "downhill onto a kept minimum");
        assert!(
            !m.offer(-9.0, -10.0 + 1e-6, None),
            "the uphill one was kept"
        );
    }

    #[test]
    fn structure_memory_tells_apart_minima_at_one_energy() {
        let mut m = CompletionMemory::new(OrbitMemory::Structure, 1e-6, 0.1);
        let a = array![1.0, 1.0, 2.0];
        let b = array![1.0, 1.5, 2.0];
        let near_a = array![1.0, 1.05, 2.0];
        m.keep(-5.0, Some(a.view()));
        assert!(m.holds(-5.0, Some(a.view())));
        assert!(m.holds(-5.0 + 5e-7, Some(near_a.view())));
        assert!(!m.holds(-5.0, Some(b.view())), "another structure");
        assert!(!m.holds(-4.0, Some(a.view())), "another energy");
        assert!(!m.holds(-5.0, None), "no descriptor");
        assert!(
            m.offer(-4.0, -5.0, Some(b.view())),
            "a new structure is fresh"
        );
        assert_eq!(m.len(), 2);
        assert!(!m.offer(-4.0, -5.0, Some(b.view())));
    }

    #[test]
    fn the_energy_tolerance_scales_with_the_energy_unit() {
        let mut m = CompletionMemory::new(OrbitMemory::Energy, 1e-6 * 1000.0, 0.7);
        m.keep(-50_000.0, None);
        assert!(m.holds(-50_000.0 + 5e-4, None));
        assert!(!m.holds(-50_000.0 + 2e-3, None));
    }
}
