//! Packing-family identity for cooperative search.
//!
//! The exact census radius is quench reproducibility. A packing family is
//! the unit mean of per-center leftover SOAP/ACE rows: isomers of one
//! funnel sit together; Marks and Mackay stay apart. No named morphology
//! enters the comparison.

use ndarray::{Array1, ArrayView1};

use crate::soap::{local_nu3_z, SoapSpec};

/// Merge radius on the unit leftover mean. Isomers of one packing sit
/// below this; the LJ75 Mackay–Marks leftover gap is 0.163.
pub const PACKING_MERGE: f64 = 0.10;

/// SOAP/ACE spec whose leftover block sees packing, not quench noise.
const PACKING_SPEC: SoapSpec = SoapSpec {
    n_max: 3,
    l_max: 6,
    rcut_nn: 3.5,
};

/// Closed-shell leftover needs a real neighbour cloud.
const MINIMUM_PACKING_ATOMS: usize = 13;

/// Unit leftover mean of a coordinate vector, if the cloud is large enough.
pub fn packing_fingerprint(coordinates: &[f64]) -> Option<Vec<f64>> {
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
    let dim = loc.ncols();
    let soap_dim = PACKING_SPEC.feat_dim(None);
    let leftover_start = soap_high_l_start(soap_dim);
    if leftover_start >= dim {
        return None;
    }
    let leftover_dim = dim - leftover_start;
    let mut mean = vec![0.0; leftover_dim];
    for i in 0..loc.nrows() {
        for t in 0..leftover_dim {
            mean[t] += loc[[i, leftover_start + t]];
        }
    }
    let n = loc.nrows() as f64;
    for value in &mut mean {
        *value /= n;
    }
    unit_in_place(&mut mean).then_some(mean)
}

/// Euclidean distance between two unit leftover means.
pub fn packing_distance(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Whether two leftover means belong to one packing family.
pub fn same_packing(left: &[f64], right: &[f64]) -> bool {
    packing_distance(left, right) <= PACKING_MERGE
}

/// Leader-clustered packing families observed by the coordinator.
#[derive(Clone, Debug, Default)]
pub struct PackingBook {
    leaders: Vec<Vec<f64>>,
    visits: Vec<u64>,
}

impl PackingBook {
    /// Record one observation into its packing family and return the family index.
    pub fn observe(&mut self, fingerprint: &[f64]) -> Option<usize> {
        if fingerprint.is_empty() {
            return None;
        }
        if let Some(index) = self.family_of(fingerprint) {
            self.visits[index] = self.visits[index].saturating_add(1);
            return Some(index);
        }
        self.leaders.push(fingerprint.to_vec());
        self.visits.push(1);
        Some(self.leaders.len() - 1)
    }

    /// Family index without mutating visit counts.
    pub fn family_of(&self, fingerprint: &[f64]) -> Option<usize> {
        self.leaders
            .iter()
            .enumerate()
            .find(|(_, leader)| same_packing(fingerprint, leader))
            .map(|(index, _)| index)
    }

    /// Exact observations assigned to one packing family.
    pub fn visits(&self, family: usize) -> u64 {
        self.visits.get(family).copied().unwrap_or(0)
    }

    /// Distance to the nearest leader of a *different* packing family.
    pub fn novelty(&self, fingerprint: &[f64]) -> f64 {
        let local = self.family_of(fingerprint);
        self.leaders
            .iter()
            .enumerate()
            .filter(|(index, _)| local != Some(*index))
            .map(|(_, leader)| packing_distance(fingerprint, leader))
            .fold(None, |nearest, distance| {
                Some(nearest.map_or(distance, |current: f64| current.min(distance)))
            })
            .unwrap_or(0.0)
    }
}

fn soap_high_l_start(soap_dim: usize) -> usize {
    let l_channels = PACKING_SPEC.l_max + 1;
    if l_channels == 0 || soap_dim < l_channels {
        return soap_dim;
    }
    let pairs = soap_dim / l_channels;
    let low = 5.min(PACKING_SPEC.l_max + 1);
    pairs * low
}

fn unit_in_place(values: &mut [f64]) -> bool {
    let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm > 0.0) || !norm.is_finite() {
        return false;
    }
    for value in values {
        *value /= norm;
    }
    true
}

/// In-crate leftover mean used by tests that do not go through a catalog.
pub fn packing_vector(coordinates: &[f64]) -> Array1<f64> {
    packing_fingerprint(coordinates).map_or_else(|| Array1::zeros(0), Array1::from)
}
