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
            self.version = self.version.wrapping_add(1);
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
        self.version = self.version.wrapping_add(1);
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

thread_local! {
    static PACKING_REFERENCES: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Publish the packings on file. Keeps the newest [`PACKING_REFERENCE_CAP`].
pub fn set_packing_references(references: Vec<Vec<f64>>) {
    PACKING_REFERENCES.with(|slot| {
        let mut held = slot.borrow_mut();
        *held = references;
        let excess = held.len().saturating_sub(PACKING_REFERENCE_CAP);
        held.drain(0..excess);
    });
}

/// Add one well to the reference cloud, and evict from the packing that can
/// spare it.
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
pub fn remember_packing_reference(coordinates: &[f64]) {
    let held = packing_references();
    let mut book = PackingBook::default();
    for reference in &held {
        book.observe(reference);
    }
    if book.observe(coordinates).is_none() {
        return;
    }
    let Some(trial) = book.histogram(coordinates) else {
        return;
    };
    let mut histograms: Vec<Vec<f64>> = Vec::with_capacity(held.len() + 1);
    for reference in &held {
        let Some(histogram) = book.histogram(reference) else {
            return;
        };
        if packing_distance(&histogram, &trial) <= PACKING_REFERENCE_MERGE {
            return;
        }
        histograms.push(histogram);
    }
    histograms.push(trial);
    let mut next = held;
    next.push(coordinates.to_vec());
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
    set_packing_references(next);
}

/// Packings on file for this replica.
pub fn packing_references() -> Vec<Vec<f64>> {
    PACKING_REFERENCES.with(|slot| slot.borrow().clone())
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
