//! Packing-family identity from per-center SOAP class histograms.
//!
//! The exact census radius is quench reproducibility. A packing family is
//! the DECAF histogram of per-atom environments, leader-clustered at a
//! fixed radius. Sealed LJ75 measurement: icosahedral reference versus
//! Marks has L1 0.69; versus the sealed ico floor the L1 is 0. No named
//! morphology enters the comparison.

use std::cell::RefCell;
use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};

use crate::soap::{SoapSpec, local_nu3_z};

/// Leader-clustering radius on per-center `local_nu3_z` rows. Same number
/// as `examples/decaf_local_classes.rs` and `rewrite_2026/data/decaf/decaf_r14.txt`.
pub const ENVIRONMENT_RADIUS: f64 = 1.4;

/// Histogram L1 at or below this is one packing family. Sits between the
/// sealed ico-floor L1 of 0 and the ico–Marks L1 of 0.69.
pub const PACKING_MERGE: f64 = 0.20;

/// DECAF used [`SoapSpec::default`], not the leftover hop spec.
const PACKING_SPEC: SoapSpec = SoapSpec {
    n_max: 3,
    l_max: 3,
    rcut_nn: 3.5,
};

/// A closed-shell class histogram needs a real neighbour cloud.
const MINIMUM_PACKING_ATOMS: usize = 13;

/// Recompute DECAF only when some atom moved more than this.
/// Sits well below [`ENVIRONMENT_RADIUS`] and the packing-family grain.
pub const PACKING_MOVE_EPS: f64 = 0.05;

/// Leader-clustered environment codebook and packing-family visits.
#[derive(Clone, Debug, Default)]
pub struct PackingBook {
    env_leaders: Vec<Vec<f64>>,
    families: Vec<Vec<f64>>,
    visits: Vec<u64>,
    /// Leftover-well arrivals per family. Hop re-observes of the same
    /// well do not increment this; that sample is 48 autocorrelated
    /// copies of the first quench and is not a packing Good--Turing
    /// draw.
    well_visits: Vec<u64>,
    histogram_cache: RefCell<Vec<CachedHistogram>>,
}

/// One remembered histogram and the path that built it.
#[derive(Clone, Debug)]
struct CachedHistogram {
    coordinates: Vec<f64>,
    histogram: Vec<f64>,
    /// Built by `assign_growing`, so every environment in it is a
    /// codebook leader. A query histogram folds unseen environments
    /// into one shared bin, which makes two structurally distinct
    /// packings look alike, and it must not credit a visit.
    grown: bool,
}

impl PackingBook {
    /// Grow the environment codebook from this structure, then count it
    /// toward a packing family.
    pub fn observe(&mut self, coordinates: &[f64]) -> Option<usize> {
        if let Some(histogram) = self.cached(coordinates, true)
            && let Some(index) = self.family_of(&histogram)
        {
            self.visits[index] = self.visits[index].saturating_add(1);
            return Some(index);
        }
        let histogram = self.assign_growing(coordinates)?;
        self.remember(coordinates, &histogram, true);
        if let Some(index) = self.family_of(&histogram) {
            self.visits[index] = self.visits[index].saturating_add(1);
            return Some(index);
        }
        self.families.push(histogram);
        self.visits.push(1);
        self.well_visits.push(0);
        Some(self.families.len() - 1)
    }

    /// Credit one leftover-SOAP well arrival to this packing family.
    pub fn credit_well(&mut self, family: usize) {
        if family >= self.families.len() {
            return;
        }
        if self.well_visits.len() < self.families.len() {
            self.well_visits.resize(self.families.len(), 0);
        }
        self.well_visits[family] = self.well_visits[family].saturating_add(1);
    }

    /// Normalized class histogram against the current codebook.
    ///
    /// Unseen environments share one extra bin so a query cannot mutate
    /// the book. Visit/offer is what grows the codebook. A structure
    /// whose atoms have not moved by [`PACKING_MOVE_EPS`] reuses the
    /// last histogram; DECAF cannot change family on that displacement.
    pub fn histogram(&self, coordinates: &[f64]) -> Option<Vec<f64>> {
        if let Some(histogram) = self.cached(coordinates, false) {
            return Some(histogram);
        }
        let histogram = self.assign_histogram(coordinates)?;
        self.remember(coordinates, &histogram, false);
        Some(histogram)
    }

    /// Remembered histogram for a structure that has not moved by
    /// [`PACKING_MOVE_EPS`]. `grown_only` restricts the answer to
    /// entries the growing path built, which is what a caller that is
    /// about to credit a visit needs.
    fn cached(&self, coordinates: &[f64], grown_only: bool) -> Option<Vec<f64>> {
        self.histogram_cache.borrow().iter().find_map(|entry| {
            if (entry.grown || !grown_only)
                && !atom_moved(&entry.coordinates, coordinates, PACKING_MOVE_EPS)
            {
                Some(entry.histogram.clone())
            } else {
                None
            }
        })
    }

    fn remember(&self, coordinates: &[f64], histogram: &[f64], grown: bool) {
        let mut cache = self.histogram_cache.borrow_mut();
        if let Some(slot) = cache
            .iter_mut()
            .find(|entry| !atom_moved(&entry.coordinates, coordinates, PACKING_MOVE_EPS))
        {
            if slot.grown && !grown {
                return;
            }
            slot.coordinates = coordinates.to_vec();
            slot.histogram = histogram.to_vec();
            slot.grown = grown;
            return;
        }
        cache.push(CachedHistogram {
            coordinates: coordinates.to_vec(),
            histogram: histogram.to_vec(),
            grown,
        });
        const CACHE_CAP: usize = 64;
        if cache.len() > CACHE_CAP {
            cache.remove(0);
        }
    }

    /// Family index without mutating visit counts.
    pub fn family_of(&self, histogram: &[f64]) -> Option<usize> {
        self.families
            .iter()
            .enumerate()
            .find(|(_, family)| same_packing(histogram, family))
            .map(|(index, _)| index)
    }

    /// Exact observations assigned to one packing family.
    pub fn visits(&self, family: usize) -> u64 {
        self.visits.get(family).copied().unwrap_or(0)
    }

    /// Occupied DECAF families on file. Visit count, not leftover-SOAP
    /// basin count. Empty until `observe` records a histogram.
    pub fn occupied_family_count(&self) -> usize {
        self.visits.iter().filter(|&&visits| visits > 0).count()
    }

    /// Histogram of each occupied packing family, in family-index order.
    ///
    /// One row per book cell with visits, not one row per live replica.
    /// The landfold floor folds these, so a packing extras have Left
    /// still sits on the map.
    pub fn occupied_histograms(&self) -> Vec<(usize, Vec<f64>)> {
        self.families
            .iter()
            .enumerate()
            .filter(|(index, _)| self.visits.get(*index).copied().unwrap_or(0) > 0)
            .map(|(index, histogram)| (index, histogram.clone()))
            .collect()
    }

    /// Chao1 completeness of leftover-well arrivals per DECAF family.
    /// Hop re-observes are not draws. Leftover SOAP still uses the
    /// unseen-mass ceiling; packing does not.
    pub fn families_saturated(&self) -> bool {
        self.well_sample().chao1_complete()
    }

    /// Leftover-well arrivals credited to packing families.
    pub fn well_sample(&self) -> GoodTuringSample {
        GoodTuringSample::from_counts(self.well_visits.iter().copied())
    }

    /// Leftover-well counts of occupied families, in family-index order.
    ///
    /// Discrete packing \(F/kT = -\ln(n/n_{\max})\) uses these, not hop
    /// re-observes.
    pub fn occupied_well_counts(&self) -> Vec<u64> {
        self.well_visits
            .iter()
            .copied()
            .filter(|&visits| visits > 0)
            .collect()
    }

    /// Distinct rematched packings among live structures.
    pub fn occupied_among<I, C>(&self, structures: I) -> usize
    where
        I: IntoIterator<Item = C>,
        C: AsRef<[f64]>,
    {
        let mut representatives: Vec<Vec<f64>> = Vec::new();
        for coordinates in structures {
            let Some(histogram) = self.histogram(coordinates.as_ref()) else {
                continue;
            };
            if self.family_of(&histogram).is_none() {
                continue;
            }
            if representatives
                .iter()
                .any(|kept| same_packing(kept, &histogram))
            {
                continue;
            }
            representatives.push(histogram);
        }
        representatives.len()
    }

    /// Family count occupancy may retire on. Leftover-SOAP Good--Turing
    /// plus two singleton DECAF slots is not two occupied funnels.
    pub fn certificate_family_count<I, C>(&self, structures: I) -> usize
    where
        I: IntoIterator<Item = C>,
        C: AsRef<[f64]>,
    {
        let rematched = self.occupied_among(structures);
        if self.families_saturated() {
            rematched
        } else {
            rematched.min(1)
        }
    }

    /// L1 to the nearest family that is not this histogram's family.
    pub fn novelty(&self, histogram: &[f64]) -> f64 {
        let local = self.family_of(histogram);
        self.families
            .iter()
            .enumerate()
            .filter(|(index, _)| local != Some(*index))
            .map(|(_, family)| packing_distance(histogram, family))
            .fold(None, |nearest, distance| {
                Some(nearest.map_or(distance, |current: f64| current.min(distance)))
            })
            .unwrap_or(0.0)
    }

    fn assign_histogram(&self, coordinates: &[f64]) -> Option<Vec<f64>> {
        if !coordinates.len().is_multiple_of(3) {
            return None;
        }
        let atoms = coordinates.len() / 3;
        if atoms < MINIMUM_PACKING_ATOMS {
            return None;
        }
        let loc = local_nu3_z(ArrayView1::from(coordinates), PACKING_SPEC, None);
        if loc.nrows() == 0 || loc.ncols() == 0 {
            return None;
        }
        let mut counts = BTreeMap::<usize, usize>::new();
        let unseen = self.env_leaders.len();
        for i in 0..loc.nrows() {
            let row = loc.row(i);
            match nearest_leader(&self.env_leaders, row.as_slice().unwrap()) {
                Some(class) => *counts.entry(class).or_insert(0) += 1,
                None => *counts.entry(unseen).or_insert(0) += 1,
            }
        }
        Some(dense_normalized(&counts, unseen + 1))
    }

    fn assign_growing(&mut self, coordinates: &[f64]) -> Option<Vec<f64>> {
        if !coordinates.len().is_multiple_of(3) {
            return None;
        }
        let atoms = coordinates.len() / 3;
        if atoms < MINIMUM_PACKING_ATOMS {
            return None;
        }
        let loc = local_nu3_z(ArrayView1::from(coordinates), PACKING_SPEC, None);
        if loc.nrows() == 0 || loc.ncols() == 0 {
            return None;
        }
        let mut counts = BTreeMap::<usize, usize>::new();
        for i in 0..loc.nrows() {
            let row = loc.row(i);
            let slice = row.as_slice().unwrap();
            let class = match nearest_leader(&self.env_leaders, slice) {
                Some(class) => class,
                None => {
                    self.env_leaders.push(slice.to_vec());
                    self.env_leaders.len() - 1
                }
            };
            *counts.entry(class).or_insert(0) += 1;
        }
        Some(dense_normalized(&counts, self.env_leaders.len()))
    }
}

fn atom_moved(left: &[f64], right: &[f64], eps: f64) -> bool {
    if left.len() != right.len() {
        return true;
    }
    let n = left.len() / 3;
    let limit = eps * eps;
    (0..n).any(|atom| {
        let d0 = left[3 * atom] - right[3 * atom];
        let d1 = left[3 * atom + 1] - right[3 * atom + 1];
        let d2 = left[3 * atom + 2] - right[3 * atom + 2];
        d0 * d0 + d1 * d1 + d2 * d2 > limit
    })
}

/// L1 distance between two normalized histograms, zero-padded to a common length.
pub fn packing_distance(left: &[f64], right: &[f64]) -> f64 {
    let n = left.len().max(right.len());
    if n == 0 {
        return f64::INFINITY;
    }
    (0..n)
        .map(|i| {
            let a = left.get(i).copied().unwrap_or(0.0);
            let b = right.get(i).copied().unwrap_or(0.0);
            (a - b).abs()
        })
        .sum()
}

/// Whether two histograms belong to one packing family.
pub fn same_packing(left: &[f64], right: &[f64]) -> bool {
    packing_distance(left, right) <= PACKING_MERGE
}

/// Good--Turing sample: `n` arrivals, `n1` singletons, `n2` doubletons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoodTuringSample {
    /// Independent draws (leftover-well arrivals, not hops).
    pub n: u64,
    /// Types seen exactly once.
    pub n1: u64,
    /// Types seen exactly twice. Chao1 uses this.
    pub n2: u64,
}

impl GoodTuringSample {
    /// Collapse a multiplicity list to `n`, `n1`, and `n2`. Zero counts drop.
    pub fn from_counts(arrivals: impl IntoIterator<Item = u64>) -> Self {
        let counts: Vec<u64> = arrivals.into_iter().filter(|&visits| visits > 0).collect();
        Self {
            n: counts.iter().copied().sum(),
            n1: counts.iter().filter(|&&visits| visits == 1).count() as u64,
            n2: counts.iter().filter(|&&visits| visits == 2).count() as u64,
        }
    }

    /// Estimated probability the next draw is a new type.
    pub fn unseen(self) -> Option<f64> {
        (self.n != 0).then(|| self.n1 as f64 / self.n as f64)
    }

    /// Chao, A. (1984), *Scand. J. Statist.* 11:265-270: unseen-family
    /// lower bound \(n_1^2/(2n_2)\).
    ///
    /// Zero when \(n_1=0\). Unbounded when \(n_1>0\) and \(n_2=0\).
    pub fn chao1_unseen(self) -> Option<f64> {
        if self.n1 == 0 {
            return Some(0.0);
        }
        if self.n2 == 0 {
            return None;
        }
        Some((self.n1 as f64) * (self.n1 as f64) / (2.0 * self.n2 as f64))
    }

    /// Chao1 completeness of the packing codebook: no singletons, so
    /// \(\hat S_{\mathrm{Chao1}}=S_{\mathrm{obs}}\). Leftover SOAP still
    /// uses [`Self::saturated`] (hatch-stable coverage).
    pub fn chao1_complete(self) -> bool {
        self.n >= crate::catalog::PRODUCTION_MINIMUM_VISITS && self.n1 == 0
    }

    /// Production floor and unseen-mass ceiling. Leftover-SOAP dwell.
    pub fn saturated(self) -> bool {
        self.n >= crate::catalog::PRODUCTION_MINIMUM_VISITS
            && self
                .unseen()
                .is_some_and(|mass| mass < crate::catalog::PRODUCTION_MAX_UNSEEN_MASS)
    }
}

/// Good--Turing on leftover-well arrivals. `n` is arrivals, not hops.
pub fn leftover_arrivals_saturated(arrivals: impl IntoIterator<Item = u64>) -> bool {
    GoodTuringSample::from_counts(arrivals).saturated()
}

/// Whether `trial` is a different DECAF packing family from `origin`.
///
/// One throwaway book: observe `origin`, then histogram `trial` against
/// that codebook. Occupancy Leave accepts a quench only when this is
/// true. Leftover-SOAP off-well is not a family change.
pub fn different_decaf_family(origin: &[f64], trial: &[f64]) -> bool {
    let mut book = PackingBook::default();
    if book.observe(origin).is_none() {
        return false;
    }
    let Some(home) = book.histogram(origin) else {
        return false;
    };
    match book.histogram(trial) {
        Some(away) => !same_packing(&home, &away),
        None => true,
    }
}

/// Dense histogram used by tests that do not hold a coordinator book.
pub fn packing_vector(coordinates: &[f64]) -> Array1<f64> {
    let mut book = PackingBook::default();
    book.assign_growing(coordinates)
        .map_or_else(|| Array1::zeros(0), Array1::from)
}

/// Histogram of one structure against a throwaway codebook. Tests that
/// compare two structures must share a book; use [`PackingBook`].
pub fn packing_fingerprint(coordinates: &[f64]) -> Option<Vec<f64>> {
    let mut book = PackingBook::default();
    book.assign_growing(coordinates)
}

fn nearest_leader(leaders: &[Vec<f64>], row: &[f64]) -> Option<usize> {
    let mut best = None;
    for (index, leader) in leaders.iter().enumerate() {
        if leader.len() != row.len() {
            continue;
        }
        let distance = leader
            .iter()
            .zip(row)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        if distance <= ENVIRONMENT_RADIUS {
            match best {
                Some((_, kept)) if kept <= distance => {}
                _ => best = Some((index, distance)),
            }
        }
    }
    best.map(|(index, _)| index)
}

fn dense_normalized(counts: &BTreeMap<usize, usize>, dim: usize) -> Vec<f64> {
    let total = counts.values().sum::<usize>().max(1) as f64;
    let mut out = vec![0.0; dim];
    for (&class, &count) in counts {
        if class < dim {
            out[class] = count as f64 / total;
        }
    }
    out
}
