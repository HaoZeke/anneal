//! Packing-family identity from per-center SOAP class histograms.
//!
//! The exact census radius is quench reproducibility. A book cell is the
//! DECAF histogram of per-atom environments, leader-clustered at a fixed
//! radius, merged at [`PACKING_MERGE`]. A packing is a single-linkage
//! community of cells at [`PACKING_LINK`], because isomers of one packing
//! spread further from their own reference than a competing packing sits
//! from it: measured on LJ75, the icosahedral shelf reaches L1 0.56 from the
//! ico reference while ico-Marks is 0.4267. No named morphology enters
//! either comparison.

use std::cell::RefCell;
use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};

use crate::soap::{SoapSpec, local_nu3_z};

/// Leader-clustering radius on per-center `local_nu3_z` rows. Same number
/// as `examples/decaf_local_classes.rs` and `rewrite_2026/data/decaf/decaf_r14.txt`.
pub const ENVIRONMENT_RADIUS: f64 = 1.4;

/// Histogram L1 at or below this is one book cell. The cell grain, not the
/// packing grain: a live LJ75 book puts tens of icosahedral cells above it.
pub const PACKING_MERGE: f64 = 0.20;

/// Single-linkage radius that separates packings on a shared codebook.
///
/// A radius from one reference cannot do this. Measured on quenched isomers
/// within \(8\varepsilon\) of each icosahedral floor, against the sealed
/// fixtures (`examples/decaf_packing_separator`): the LJ75 shelf reaches L1
/// \(0.56\) from the ico reference while ico-Marks is \(0.4267\), so the
/// shelf spread straddles the gap and no ball around ico holds one and not
/// the other. Single linkage separates them instead.
///
/// The radius is the same measurement on both sizes. LJ75, 69 shelf isomers
/// plus both fixtures: Marks stands alone from \(0.10\) through
/// \(0.40\) while the shelf chains into one component of 64 to 70, and at
/// \(0.45\) the shelf reaches Marks. LJ38, 154 shelf isomers, where ico-Oh
/// is \(1.1579\): Oh stands alone throughout, and the shelf chains 58 of
/// 154 at \(0.20\), 148 at \(0.30\), 153 at \(0.35\), 155 at
/// \(0.40\). Below \(0.30\) a smaller cluster over-splits, because one
/// atom carries \(2/N\) of the histogram; above \(0.40\) the LJ75 shelf
/// swallows Marks. This is the middle of what both sizes allow.
pub const PACKING_LINK: f64 = 0.35;

/// DECAF used [`SoapSpec::default`], not the leftover hop spec.
pub const PACKING_SPEC: SoapSpec = SoapSpec {
    n_max: 3,
    l_max: 3,
    rcut_nn: 3.5,
};

/// A closed-shell class histogram needs a real neighbour cloud.
pub(crate) const MINIMUM_PACKING_ATOMS: usize = 13;

/// Recompute DECAF only when some atom moved more than this.
/// Sits well below [`ENVIRONMENT_RADIUS`] and the packing-family grain.
pub const PACKING_MOVE_EPS: f64 = 0.05;

/// Leader-clustered environment codebook and packing-family visits.
#[derive(Clone, Debug, Default)]
pub struct PackingBook {
    env_leaders: Vec<Vec<f64>>,
    families: Vec<Vec<f64>>,
    visits: Vec<u64>,
    /// Bumped whenever a cell or a well arrival changes the book.
    ///
    /// Folding the book is single linkage over its cells, quadratic in
    /// their number, and the policy response asks for the fold several
    /// times per request. A coordinator serving 48 replicas spends its
    /// core there rather than on the replicas. The version lets a caller
    /// keep the fold it already computed while the book is unchanged.
    version: u64,
    /// Leftover-well arrivals per family. Hop re-observes of the same
    /// well do not increment this; that sample is 48 autocorrelated
    /// copies of the first quench and is not a packing Good--Turing
    /// draw.
    well_visits: Vec<u64>,
    histogram_cache: RefCell<Vec<CachedHistogram>>,
    /// Incremental single-linkage parents of [`Self::families`] at
    /// [`PACKING_LINK`]. A new cell unions against the book once; visits
    /// do not rebuild the whole fold.
    community_parent: Vec<usize>,
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
    /// Changes to the book since it was created.
    pub fn version(&self) -> u64 {
        self.version
    }

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
            self.version = self.version.wrapping_add(1);
            return Some(index);
        }
        self.families.push(histogram);
        self.visits.push(1);
        self.well_visits.push(0);
        let index = self.families.len() - 1;
        self.community_parent.push(index);
        self.union_new_cell(index);
        self.version = self.version.wrapping_add(1);
        Some(index)
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
        self.version = self.version.wrapping_add(1);
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

    /// Occupied DECAF cells on file. Visit count, not leftover-SOAP
    /// basin count. Empty until `observe` records a histogram.
    ///
    /// A live LJ75 icosahedral shelf holds tens of these. Packing
    /// identity is [`Self::occupied_packing_count`].
    pub fn occupied_family_count(&self) -> usize {
        self.visits.iter().filter(|&&visits| visits > 0).count()
    }

    /// Occupied packing communities on file, at [`PACKING_LINK`].
    ///
    /// Cell count is not this: isomers of one packing open many cells,
    /// and treating each as a family sends extras between icosahedral
    /// wells that are not a second funnel.
    pub fn occupied_packing_count(&self) -> usize {
        self.occupied_community_labels()
            .1
            .into_iter()
            .max()
            .map_or(0, |last| last + 1)
    }

    fn find_community(&self, mut node: usize) -> usize {
        let n = self.community_parent.len();
        if n == 0 || node >= n {
            return node;
        }
        while self.community_parent[node] != node {
            node = self.community_parent[node];
        }
        node
    }

    fn union_new_cell(&mut self, index: usize) {
        for other in 0..index {
            if self.visits.get(other).copied().unwrap_or(0) == 0 {
                continue;
            }
            if packing_distance(&self.families[index], &self.families[other]) > PACKING_LINK {
                continue;
            }
            let a = self.find_community(index);
            let b = self.find_community(other);
            if a != b {
                self.community_parent[a] = b;
            }
        }
    }

    /// Occupied cells and their incremental packing-community labels.
    pub fn occupied_community_labels(&self) -> (Vec<(usize, Vec<f64>)>, Vec<usize>) {
        let occupied = self.occupied_histograms();
        let raw: Vec<usize> = occupied
            .iter()
            .map(|(index, _)| self.find_community(*index))
            .collect();
        let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
        let labels = raw
            .into_iter()
            .map(|root| {
                let next = seen.len();
                *seen.entry(root).or_insert(next)
            })
            .collect();
        (occupied, labels)
    }

    /// Family indices in the same packing community as `query`.
    ///
    /// The book fold is cached. The query is assigned by one single-linkage
    /// step onto that fold: a cell at or below [`PACKING_LINK`] inherits
    /// its community. A query that does not chain is not a new fold.
    pub fn families_sharing_community(&self, query: &[f64]) -> Vec<usize> {
        let (occupied, labels) = self.occupied_community_labels();
        let Some(label) = occupied.iter().enumerate().find_map(|(i, (_, histogram))| {
            (packing_distance(query, histogram) <= PACKING_LINK).then_some(labels[i])
        }) else {
            return Vec::new();
        };
        occupied
            .iter()
            .enumerate()
            .filter(|(i, _)| labels[*i] == label)
            .map(|(_, (index, _))| *index)
            .collect()
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

    /// Leftover-well arrivals credited to one packing family.
    pub fn well_visits_of(&self, family: usize) -> u64 {
        self.well_visits.get(family).copied().unwrap_or(0)
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

    /// Distinct packings among live structures, at [`PACKING_LINK`].
    ///
    /// [`Self::occupied_among`] counts book cells, and isomers of one packing
    /// hold tens of them, so counting cells against a packing floor certifies
    /// a second funnel that is not there.
    pub fn occupied_packings_among<I, C>(&self, structures: I) -> usize
    where
        I: IntoIterator<Item = C>,
        C: AsRef<[f64]>,
    {
        let mut histograms: Vec<Vec<f64>> = Vec::new();
        for coordinates in structures {
            let Some(histogram) = self.histogram(coordinates.as_ref()) else {
                continue;
            };
            if self.family_of(&histogram).is_none() {
                continue;
            }
            histograms.push(histogram);
        }
        packing_community_count(&histograms)
    }

    /// Family count occupancy may retire on. Leftover-SOAP Good--Turing
    /// plus two singleton DECAF slots is not two occupied funnels.
    pub fn certificate_family_count<I, C>(&self, structures: I) -> usize
    where
        I: IntoIterator<Item = C>,
        C: AsRef<[f64]>,
    {
        let rematched = self.occupied_packings_among(structures);
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

    /// Share of this structure's atoms whose local environment matches
    /// no cell the book has grown: census surprisal at the cell grain,
    /// straight from the descriptor.
    ///
    /// The packing gap sums how many atoms changed class and discards
    /// which classes, so a crossing into genuinely new environments and
    /// a shuffle among the shelf's old ones read alike. This is the
    /// half the gap discards, computed by the machinery the book
    /// already runs: the query histogram folds unseen environments into
    /// one bin, and the bin's mass is the answer. No template, no
    /// prototype, no reference structure -- the codebook is the run's
    /// own history, which is what makes it a coordinate any system
    /// carries.
    pub fn unseen_share(&self, coordinates: &[f64]) -> Option<f64> {
        let histogram = self.assign_histogram(coordinates)?;
        histogram.last().copied()
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

/// Per-atom DECAF class under a throwaway leader book of this structure.
///
/// Lai, Poths, Matera, Scheurer and Reuter (*Phys. Rev. Lett.* **134**,
/// 096201 (2025)): APE classifies local environments and queues dimer
/// searches on a highlighted atom, not a global packing-mean cover.
pub fn atom_decaf_classes(coordinates: &[f64]) -> Vec<usize> {
    if !coordinates.len().is_multiple_of(3) {
        return Vec::new();
    }
    let loc = local_nu3_z(ArrayView1::from(coordinates), PACKING_SPEC, None);
    let n_at = loc.nrows();
    if n_at == 0 {
        return Vec::new();
    }
    let mut leaders: Vec<Vec<f64>> = Vec::new();
    let mut classes = vec![0usize; n_at];
    for i in 0..n_at {
        let row = loc.row(i).to_vec();
        match nearest_leader(&leaders, &row) {
            Some(class) => classes[i] = class,
            None => {
                leaders.push(row);
                classes[i] = leaders.len() - 1;
            }
        }
    }
    classes
}

/// APE to-do list: atoms grouped by DECAF class, most occupied class first.
///
/// One queue entry per local environment class. The extra walks this
/// list; each Leave highlights one atom and seeds a dimer there.
pub fn ape_highlight_queue(coordinates: &[f64]) -> Vec<(usize, usize)> {
    let classes = atom_decaf_classes(coordinates);
    if classes.is_empty() {
        return Vec::new();
    }
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (atom, class) in classes.iter().copied().enumerate() {
        buckets.entry(class).or_default().push(atom);
    }
    let mut order: Vec<(usize, Vec<usize>)> = buckets.into_iter().collect();
    order.sort_by(|left, right| {
        right
            .1
            .len()
            .cmp(&left.1.len())
            .then_with(|| left.0.cmp(&right.0))
    });
    order
        .into_iter()
        .flat_map(|(class, atoms)| atoms.into_iter().map(move |atom| (atom, class)))
        .collect()
}

/// APE seed: displace the highlighted atom along a 3-space cover.
pub fn ape_local_seed(coordinates: &[f64], atom: usize, amplitude: f64, draw: usize) -> Vec<f64> {
    let mut out = coordinates.to_vec();
    let n_at = out.len() / 3;
    if atom >= n_at || !(amplitude.is_finite() && amplitude > 0.0) {
        return out;
    }
    let direction = crate::hypersphere::cover_direction(8, 3, draw % 8);
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return out;
    }
    let scale = amplitude / norm;
    for k in 0..3 {
        out[3 * atom + k] += scale * direction[k];
    }
    out
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

/// Single-linkage community labels at `radius`, in first-appearance order.
///
/// Packing identity is a community of book cells, not a ball around one
/// reference: quenched isomers of one packing spread further from their own
/// reference than a competing packing sits from it. Chaining through the
/// cells the book already holds is what keeps the isomer shelf together and
/// still leaves a genuinely different packing on its own.
pub fn packing_link_labels(histograms: &[Vec<f64>], radius: f64) -> Vec<usize> {
    let n = histograms.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if packing_distance(&histograms[i], &histograms[j]) <= radius {
                let a = find(&mut parent, i);
                let b = find(&mut parent, j);
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    (0..n)
        .map(|i| {
            let root = find(&mut parent, i);
            let next = seen.len();
            *seen.entry(root).or_insert(next)
        })
        .collect()
}

/// [`packing_link_labels`] at [`PACKING_LINK`].
pub fn packing_communities(histograms: &[Vec<f64>]) -> Vec<usize> {
    packing_link_labels(histograms, PACKING_LINK)
}

/// Distinct packings among `histograms`.
pub fn packing_community_count(histograms: &[Vec<f64>]) -> usize {
    packing_communities(histograms)
        .into_iter()
        .max()
        .map_or(0, |last| last + 1)
}

/// Packings on file, as coordinates, for a replica with no codebook.
///
/// Histograms are only comparable inside one codebook, so what crosses to a
/// replica is structures. The replica grows a throwaway book over the
/// references plus its own pair and reads the communities off that.
///
/// One structure per packing is not enough. A packing is what its cells
/// chain into, and a quenched LJ75 icosahedral isomer four to eight
/// \(\varepsilon\) above the shelf floor sits up to L1 0.56 from the ico
/// reference: against that reference alone it reads as a packing of its own,
/// and against the shelf it chains straight back. So the references are a
/// cloud of cells, deduplicated at the cell grain.
const PACKING_REFERENCE_CAP: usize = 64;

/// Histogram L1 below which two references are the same well.
///
/// The cloud is what the Leave invert repels a quench from, so what it
/// must hold is where the other chains *are*, not how many packings they
/// amount to. Deduplicating at the cell grain collapses an ensemble
/// sitting in one funnel to a single entry: measured on LJ75 with the
/// cross-chain feed live, the cloud held exactly one reference at 12, 24
/// and 48 chains alike, so the chains could not push each other apart
/// during minimisation however many of them there were. Only a structure
/// that is numerically the same well is dropped.
pub const PACKING_REFERENCE_MERGE: f64 = 1e-9;

/// Wells a chain draws from the shared catalog each time it arms a Leave.
///
/// The cloud is the only place chains meet during a quench, and it was fed
/// one structure per checkpoint. That rate does not depend on how many
/// chains are running, so the bias a chain feels is the same on
/// forty-eight chains as on one and the ensemble buys nothing at the
/// minimisation level. The catalog already holds what the others are
/// standing on, and its size does grow with the ensemble, so an arm takes
/// several entries at once and the repulsion scales with chain count.
/// Twelve fills the [`PACKING_REFERENCE_CAP`] within a handful of Leaves
/// while staying well under the round trips a checkpoint can afford.
pub const PACKING_REFERENCE_DRAWS: usize = 12;

/// One well the Leave repels a quench from.
///
/// The coordinates are what the invert needs. The other two fields are
/// what tells a run how *expensive* the well is rather than how deep it
/// is: a packing the ensemble keeps arriving on is entropically
/// stabilised, and its free-energy depth exceeds its potential depth by
/// the temperature times the log of its arrivals.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackingReference {
    /// Quenched structure.
    pub coordinates: Vec<f64>,
    /// Arrivals on this well, never zero once it is on file. The
    /// configurational entropy of the packing is the log of this.
    pub visits: u32,
    /// Bias already deposited here, so a later hill can be scaled down by
    /// what is already standing.
    pub deposit: f64,
}

thread_local! {
    static PACKING_REFERENCES: RefCell<Vec<PackingReference>> = const { RefCell::new(Vec::new()) };
}

/// Publish the packings on file. Keeps the newest [`PACKING_REFERENCE_CAP`].
///
/// Arrival counts start at one: a well on file has been arrived on once
/// by construction, and a zero count would read as an impossible packing
/// rather than as a rare one.
pub fn set_packing_references(references: Vec<Vec<f64>>) {
    PACKING_REFERENCES.with(|slot| {
        let mut held = slot.borrow_mut();
        *held = references
            .into_iter()
            .map(|coordinates| PackingReference {
                coordinates,
                visits: 1,
                deposit: 0.0,
            })
            .collect();
        let excess = held.len().saturating_sub(PACKING_REFERENCE_CAP);
        held.drain(0..excess);
    });
}

/// The cloud with its arrival counts and standing bias.
pub fn packing_reference_book() -> Vec<PackingReference> {
    PACKING_REFERENCES.with(|slot| slot.borrow().clone())
}

/// Record that `amount` of bias now stands on the well nearest
/// `coordinates`, so the next hill there can be scaled by what is
/// already deposited.
pub fn credit_packing_deposit(coordinates: &[f64], amount: f64) {
    if !amount.is_finite() || amount <= 0.0 {
        return;
    }
    let held = packing_references();
    let mut book = PackingBook::default();
    for reference in &held {
        book.observe(reference);
    }
    let Some(trial) = book.histogram(coordinates) else {
        return;
    };
    let mut nearest = None;
    let mut best = f64::INFINITY;
    for (index, reference) in held.iter().enumerate() {
        let Some(histogram) = book.histogram(reference) else {
            continue;
        };
        let distance = packing_distance(&histogram, &trial);
        if distance < best {
            best = distance;
            nearest = Some(index);
        }
    }
    let Some(index) = nearest else {
        return;
    };
    PACKING_REFERENCES.with(|slot| {
        if let Some(reference) = slot.borrow_mut().get_mut(index) {
            reference.deposit += amount;
        }
    });
}

/// Include one remotely observed well in the reference cloud without
/// crediting another arrival, and evict from the packing that can spare it.
///
/// The cloud is what the Leave invert repels a quench from, so it holds
/// wells rather than packings: an ensemble of 48 chains in one funnel has
/// to push against 48 positions, not against the single packing they
/// collectively occupy. Only a numerically identical well is dropped, at
/// [`PACKING_REFERENCE_MERGE`].
///
/// Over the cap the oldest member of the largest community goes, so a
/// community with one member survives: it is the only record that packing
/// exists, and dropping it would let the run rediscover it as new.
pub fn include_packing_reference(coordinates: &[f64]) {
    update_packing_reference(coordinates, false);
}

/// Record one local arrival in the reference cloud.
///
/// A matching well gains one configurational-entropy visit. A novel well is
/// included with its first arrival and follows the same bounded-cloud policy
/// as [`include_packing_reference`].
pub fn remember_packing_reference(coordinates: &[f64]) {
    update_packing_reference(coordinates, true);
}

fn update_packing_reference(coordinates: &[f64], count_matching_arrival: bool) {
    let held = packing_reference_book();
    let mut book = PackingBook::default();
    for reference in &held {
        book.observe(&reference.coordinates);
    }
    if book.observe(coordinates).is_none() {
        return;
    }
    let Some(trial) = book.histogram(coordinates) else {
        return;
    };
    let mut histograms: Vec<Vec<f64>> = Vec::with_capacity(held.len() + 1);
    for (index, reference) in held.iter().enumerate() {
        let Some(histogram) = book.histogram(&reference.coordinates) else {
            return;
        };
        if packing_distance(&histogram, &trial) <= PACKING_REFERENCE_MERGE {
            // Arriving again on a well already on file is the entropy
            // measurement, not a duplicate to discard. A packing the run
            // keeps landing in is one with many ways to be reached, and
            // the count of those arrivals is the configurational entropy
            // the free-energy deposit needs. Discarding it left the cloud
            // honest about where the ensemble is and silent about how
            // much of it is there.
            if count_matching_arrival {
                PACKING_REFERENCES.with(|slot| {
                    if let Some(reference) = slot.borrow_mut().get_mut(index) {
                        reference.visits = reference.visits.saturating_add(1);
                    }
                });
            }
            return;
        }
        histograms.push(histogram);
    }
    histograms.push(trial);
    let mut next = held;
    next.push(PackingReference {
        coordinates: coordinates.to_vec(),
        visits: 1,
        deposit: 0.0,
    });
    while next.len() > PACKING_REFERENCE_CAP {
        let labels = packing_communities(&histograms);
        let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
        for &label in &labels {
            *sizes.entry(label).or_insert(0) += 1;
        }
        let Some((&largest, _)) = sizes.iter().max_by_key(|(_, size)| **size) else {
            break;
        };
        let Some(evict) = labels.iter().position(|&label| label == largest) else {
            break;
        };
        next.remove(evict);
        histograms.remove(evict);
    }
    PACKING_REFERENCES.with(|slot| *slot.borrow_mut() = next);
}

/// Packings on file for this replica, as structures.
pub fn packing_references() -> Vec<Vec<f64>> {
    PACKING_REFERENCES.with(|slot| {
        slot.borrow()
            .iter()
            .map(|reference| reference.coordinates.clone())
            .collect()
    })
}

/// Whether `trial` sits in a packing that `origin` and `references` do not.
///
/// One throwaway book over every structure, single linkage at
/// [`PACKING_LINK`], then the community of the trial against the community of
/// the origin. With no references this is the pairwise form: the trial does
/// not chain to the origin.
pub fn leaves_packing(origin: &[f64], trial: &[f64], references: &[Vec<f64>]) -> bool {
    let mut book = PackingBook::default();
    if book.observe(origin).is_none() {
        return false;
    }
    for reference in references {
        book.observe(reference);
    }
    if book.observe(trial).is_none() {
        return false;
    }
    let Some(home) = book.histogram(origin) else {
        return false;
    };
    let Some(away) = book.histogram(trial) else {
        return true;
    };
    let mut histograms = vec![home, away];
    for reference in references {
        if let Some(histogram) = book.histogram(reference) {
            histograms.push(histogram);
        }
    }
    let labels = packing_communities(&histograms);
    labels[1] != labels[0]
}

/// [`leaves_packing`] against the packings this replica holds on file.
pub fn different_packing_family(origin: &[f64], trial: &[f64]) -> bool {
    leaves_packing(origin, trial, &packing_references())
}

/// Catalog slots drawn at a checkpoint to find nearby chains.
///
/// The invert is not a per-hop all-to-all. A checkpoint already talks
/// to the catalog; these draws are the only extra mail, and only a
/// neighbour in the map is armed.
pub const INVERT_NEIGHBOR_DRAWS: usize = 4;

/// Whether two structures sit on the same side of the packing map.
///
/// Pairwise L1 at [`PACKING_LINK`], no chaining through a third well.
/// A far packing is a hear, not an invert neighbour. The predicate
/// names no morphology.
pub fn nearby_packing(here: &[f64], other: &[f64]) -> bool {
    if here.len() != other.len() || here.is_empty() {
        return false;
    }
    let mut book = PackingBook::default();
    book.observe(here);
    book.observe(other);
    match (book.histogram(here), book.histogram(other)) {
        (Some(a), Some(b)) => packing_distance(&a, &b) <= PACKING_LINK,
        _ => false,
    }
}

/// Reference-cloud entries that are invert neighbours of `here`.
pub fn nearby_packing_book(here: &[f64]) -> Vec<PackingReference> {
    packing_reference_book()
        .into_iter()
        .filter(|reference| nearby_packing(here, &reference.coordinates))
        .collect()
}

/// Paving pile kept per packing community rather than per basin.
///
/// The hop acceptance already walks a biased landscape, but the pile it
/// reads is [`crate::bias::BasinBias`], keyed on the cluster fingerprint.
/// Every icosahedral isomer therefore opens its own account, and the
/// icosahedral shelf of LJ75 carries hundreds of minima that the live
/// book splits into about thirty-two DECAF families. A deposit of 0.25
/// eps a visit, fragmented that far, never approaches the 8.69 eps
/// between the funnels: the mechanism that could make the funnel
/// expensive never sees it as one place.
///
/// Nor can a single walk leave it. Measured from the sealed icosahedral
/// minimum, twenty-four raw quenches dropped along a transformed
/// trajectory reaching 1.87 in DECAF distance, against a Marks
/// separation of 0.4267, every one returned to the floor. What does
/// reach Marks is plain hopping, over a sequence of ordinary moves. So
/// the pile has to tilt the hopping, and to do that it has to be keyed
/// at the grain where icosahedral is one packing.
///
/// The cost is a map lookup per hop. A basin's community is decided once,
/// the first time that basin is seen, by the same single linkage the book
/// uses; afterwards the hop reads the pile by basin id.
#[derive(Debug, Default, Clone)]
pub struct PackingPave {
    community_of: BTreeMap<u64, usize>,
    representatives: Vec<Vec<f64>>,
    pile: Vec<f64>,
    visits: Vec<u64>,
}

impl PackingPave {
    /// An empty pile.
    pub fn new() -> Self {
        Self::default()
    }

    /// Communities opened so far.
    pub fn communities(&self) -> usize {
        self.pile.len()
    }

    /// The community `basin` belongs to, deciding it if this is the first
    /// time that basin has been seen.
    ///
    /// A structure joins the first community whose representative it does
    /// not leave, which is single linkage at [`PACKING_LINK`] against one
    /// member. That is the same rule [`leaves_packing`] applies and the
    /// same one the book folds with.
    pub fn community(&mut self, basin: u64, coordinates: &[f64]) -> usize {
        if let Some(found) = self.community_of.get(&basin) {
            return *found;
        }
        let mut joined = None;
        for (index, representative) in self.representatives.iter().enumerate() {
            if representative.len() == coordinates.len()
                && !leaves_packing(representative, coordinates, &[])
            {
                joined = Some(index);
                break;
            }
        }
        let index = joined.unwrap_or_else(|| {
            self.representatives.push(coordinates.to_vec());
            self.pile.push(0.0);
            self.visits.push(0);
            self.pile.len() - 1
        });
        self.community_of.insert(basin, index);
        index
    }

    /// Standing bias on the community holding `basin`, or zero for a
    /// basin never seen.
    pub fn potential(&self, basin: u64) -> f64 {
        self.community_of
            .get(&basin)
            .and_then(|index| self.pile.get(*index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Add one arrival on the community holding `basin`.
    ///
    /// The increment is \\(w_0+T\\ln n\\), the fixed height plus the
    /// configurational entropy of a packing reached \\(n\\) ways, scaled
    /// by \\(e^{-V/((\\gamma-1)T)}\\) so the pile converges on
    /// \\(-\\frac{\\gamma-1}{\\gamma}F\\) rather than growing without
    /// bound. That is the rule [`crate::bias::WellTemperedBias`] already
    /// deposits by, with the entropic term the fixed height leaves out.
    pub fn deposit(&mut self, basin: u64, coordinates: &[f64], w0: f64, gamma: f64, temp: f64) {
        let index = self.community(basin, coordinates);
        let Some(standing) = self.pile.get(index).copied() else {
            return;
        };
        let arrivals = self.visits.get(index).copied().unwrap_or(0);
        let entropy = if temp > 0.0 && arrivals > 0 {
            temp * (arrivals as f64).ln()
        } else {
            0.0
        };
        let taper = if gamma > 1.0 && temp > 0.0 {
            (-standing / ((gamma - 1.0) * temp)).exp()
        } else {
            1.0
        };
        let increment = (w0.max(0.0) + entropy) * taper;
        if increment.is_finite() {
            self.pile[index] = standing + increment;
        }
        if let Some(count) = self.visits.get_mut(index) {
            *count = count.saturating_add(1);
        }
    }

    /// Arrivals recorded per community, for the census and the report.
    pub fn arrivals(&self) -> &[u64] {
        &self.visits
    }

    /// Standing pile per community.
    pub fn piles(&self) -> &[f64] {
        &self.pile
    }
}

/// Shared frontier along the seam out of the occupied packing.
///
/// The crossing to another packing is a staged rare event: measured on
/// LJ75, no single move leaves the icosahedral community, and the chains
/// that reach the Marks decahedron pass it as a sequence of about ten
/// thousand ordinary accepted hops. A chain that reaches a partly
/// converted intermediate and falls back has learned where the seam is,
/// and throwing that structure away makes every other chain start the
/// climb from the floor again: n chains that do not share partial
/// progress deliver expected frontier mass \\(n\\,p^k\\) over a
/// \\(k\\)-stage road, exponentially small in the depth.
///
/// The bank keeps the lowest-energy representative in each band of DECAF
/// distance from the run's own floor. A stuck chain restarts from the
/// furthest banked structure instead of from the floor, which restores
/// each occupied stage to full population before the next attempt and
/// holds the frontier arrival rate at \\(n\\,p\\) per stage, constant in
/// the depth (`Hop.cloning_dominates` in
/// `proofs/lean/Hop/SeamLadder.lean`). Nothing here touches the
/// acceptance rule: geometry, quench and Metropolis stay raw, which is
/// what `Hop.road_priced` requires -- a penalty standing on the road
/// multiplies every stage and can only lose, and 0 of 16 against 2 of 16
/// on paired seeds is that theorem measured.
#[derive(Debug, Default, Clone)]
pub struct SeamBank {
    /// Lowest-energy representative per band, `(energy, structure)`.
    bins: Vec<Option<(f64, Vec<f64>)>>,
}

/// Gap past which an accepted structure is doorway-shaped.
///
/// Measured on the traced winning seed: the crossing was one 0.7467
/// fluctuation, past the grain and past the Marks separation, while
/// ordinary shelf wandering stays under about 0.3. The threshold sits
/// above the wandering and below the measured doorway.
pub const SEAM_DOORWAY_GAP: f64 = 0.45;

/// Energy above the floor past which a far structure is a melt, not a
/// doorway. The traced doorway sat 1.86 eps above the floor of its
/// moment; the median structure past the doorway gap sat twelve above.
/// Six keeps every doorway-like record seen and rejects the melts.
pub const SEAM_DOORWAY_WINDOW: f64 = 6.0;

/// Attempts spent from a doorway while the fluctuation is hot.
///
/// The burst is `Hop.sharedRound` with `n` attempts against one live
/// structure: the traced conversion took one hop, so the round
/// probability is large exactly here, and the burst multiplies attempts
/// at the only moment that is true. A stale restart measured 0 of 16;
/// the difference between the two is the whole finding.
pub const SEAM_BURST_SHOTS: usize = 24;

/// Width of one seam band, in the DECAF L1 the grain is quoted in.
pub const SEAM_BIN_WIDTH: f64 = 0.05;

/// Bands kept: up to 0.60, past the 0.35 grain and the measured 0.4267
/// icosahedral-to-Marks separation.
pub const SEAM_BINS: usize = 12;

/// Energy above the incumbent floor past which a structure is a melt
/// rather than a stage of the road. The ico-Marks saddles sit 8.69 and
/// 7.48 eps above the shelf; a 12 eps window keeps every on-road
/// intermediate and rejects the crushed geometries the old 0.35
/// Cartesian kick produced at +30 and worse.
pub const SEAM_WINDOW: f64 = 12.0;

impl SeamBank {
    /// An empty bank.
    pub fn new() -> Self {
        Self {
            bins: vec![None; SEAM_BINS],
        }
    }

    /// The band a gap falls in, `None` below the first rung.
    pub fn bin_of(gap: f64) -> Option<usize> {
        if !gap.is_finite() || gap < SEAM_BIN_WIDTH {
            return None;
        }
        Some((((gap / SEAM_BIN_WIDTH) as usize).saturating_sub(1)).min(SEAM_BINS - 1))
    }

    /// Offer one accepted, quenched structure at `gap` from the floor.
    /// Returns whether it advanced or improved the frontier.
    pub fn offer(&mut self, gap: f64, energy: f64, coordinates: &[f64]) -> bool {
        if !energy.is_finite() || coordinates.is_empty() {
            return false;
        }
        let Some(bin) = Self::bin_of(gap) else {
            return false;
        };
        match &self.bins[bin] {
            Some((held, _)) if *held <= energy => false,
            _ => {
                self.bins[bin] = Some((energy, coordinates.to_vec()));
                true
            }
        }
    }

    /// The furthest banked structure, with its band index.
    pub fn frontier(&self) -> Option<(usize, f64, &[f64])> {
        self.bins.iter().enumerate().rev().find_map(|(bin, held)| {
            held.as_ref()
                .map(|(energy, coords)| (bin, *energy, coords.as_slice()))
        })
    }

    /// Bands currently holding a representative.
    pub fn occupied(&self) -> usize {
        self.bins.iter().filter(|held| held.is_some()).count()
    }

    /// A restart point: the frontier with weight `1 - epsilon`, otherwise
    /// one of the other occupied bands, chosen by the unit draw `u`.
    ///
    /// Restarting only from the top band trusts one structure, and a
    /// single poisoned representative -- a banked geometry whose forward
    /// probability is zero -- stalls the ladder with any number of
    /// episodes left. Splitting the restart keeps the advance probability
    /// positive whenever any banked band can still move, which is
    /// `Hop.eps_greedy_positive`: the frontier term may contribute
    /// nothing and the mix still advances.
    pub fn restart(&self, epsilon: f64, u: f64) -> Option<(usize, f64, &[f64])> {
        let occupied: Vec<usize> = self
            .bins
            .iter()
            .enumerate()
            .filter_map(|(bin, held)| held.as_ref().map(|_| bin))
            .collect();
        let top = *occupied.last()?;
        let pick = if occupied.len() > 1 && u < epsilon.clamp(0.0, 1.0) {
            // The draw is reused inside the epsilon slice, so the band
            // choice is uniform over the non-frontier bands without a
            // second random number.
            let others = occupied.len() - 1;
            let slot = ((u / epsilon.clamp(f64::MIN_POSITIVE, 1.0)) * others as f64) as usize;
            occupied[slot.min(others - 1)]
        } else {
            top
        };
        self.bins[pick]
            .as_ref()
            .map(|(energy, coords)| (pick, *energy, coords.as_slice()))
    }
}

/// DECAF distance from the run's floor to a trial, in the L1 the grain
/// and the seam bands are quoted in. `NaN` when either histogram cannot
/// be built, which no band accepts.
pub fn packing_seam_gap(floor: &[f64], trial: &[f64]) -> f64 {
    let mut book = PackingBook::default();
    book.observe(floor);
    book.observe(trial);
    match (book.histogram(floor), book.histogram(trial)) {
        (Some(a), Some(b)) => packing_distance(&a, &b),
        _ => f64::NAN,
    }
}

thread_local! {
    static SCREEN_KNOWN: RefCell<Vec<(f64, Vec<f64>)>> = const { RefCell::new(Vec::new()) };
}

thread_local! {
    static FRONTIER_OUT: RefCell<Vec<(f64, f64, Vec<f64>)>> = const { RefCell::new(Vec::new()) };
    static FRONTIER_IN: RefCell<Vec<(f64, f64, Vec<f64>)>> = const { RefCell::new(Vec::new()) };
}

/// Queue one raw excursion state for the shared frontier ladder. The
/// hop loop calls this at the moment a doorway-shaped structure is
/// accepted; the exchange layer drains and ships at wave cadence.
pub fn offer_frontier_post(gap: f64, energy: f64, coordinates: &[f64]) {
    if !gap.is_finite() || !energy.is_finite() || coordinates.is_empty() {
        return;
    }
    FRONTIER_OUT.with(|slot| {
        let mut held = slot.borrow_mut();
        if held.len() < 64 {
            held.push((gap, energy, coordinates.to_vec()));
        }
    });
}

/// Drain the queued outgoing frontier posts for shipping.
pub fn take_frontier_posts() -> Vec<(f64, f64, Vec<f64>)> {
    FRONTIER_OUT.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}

/// Hand the hop loop one frontier post another chain walked to. The
/// exchange layer calls this on arrivals; the run loop folds them into
/// its seam ladder, which is how the ensemble's population reaches
/// every stage of the road (`Hop.cloning_dominates`).
pub fn deliver_frontier_post(gap: f64, energy: f64, coordinates: Vec<f64>) {
    if !gap.is_finite() || !energy.is_finite() || coordinates.is_empty() {
        return;
    }
    FRONTIER_IN.with(|slot| {
        let mut held = slot.borrow_mut();
        if held.len() < 64 {
            held.push((gap, energy, coordinates));
        }
    });
}

/// Drain the arrived frontier posts since the last call.
pub fn drain_frontier_arrivals() -> Vec<(f64, f64, Vec<f64>)> {
    FRONTIER_IN.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}

/// Hand the hop loop a minimum some chain has already validated, so its
/// return screen can recognise a descent into that basin and refund the
/// rest of the relaxation.
///
/// The reallocation bound closes every channel about where the crossing
/// is; this is the channel about where the chains have been, which they
/// know perfectly. The screen already stops trials returning to the
/// chain's own position at a measured 37 force calls per hop against
/// 228 unscreened; sharing widens the recognised set, and recognition
/// is monotone in it -- `Hop.shared_recognition_cheapens` -- while the
/// acceptance still sees a real quenched energy for the basin, so the
/// per-attempt rate is untouched.
pub fn offer_known_minimum(energy: f64, coordinates: &[f64]) {
    if !energy.is_finite() || coordinates.is_empty() {
        return;
    }
    SCREEN_KNOWN.with(|slot| {
        let mut held = slot.borrow_mut();
        if held.len() < 512 {
            held.push((energy, coordinates.to_vec()));
        }
    });
}

/// Drain the minima offered since the last call, for the run loop to
/// fold into its screen bank at its own cadence.
pub fn take_known_minima() -> Vec<(f64, Vec<f64>)> {
    SCREEN_KNOWN.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}
