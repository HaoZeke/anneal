//! Invariant descriptors of rigid bodies represented by centers and rotations.
//!
//! The molecular geometry projects generalized coordinates onto actual sites.
//! Species-separated distance spectra retain orientation information without
//! treating equivalent atoms or equivalent rotation charts as distinct states.

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};

/// Site geometry relative to the center used by the objective's rigid-body chart.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RigidBodyGeometry {
    sites: Vec<[f64; 3]>,
    species: Vec<u32>,
}

impl RigidBodyGeometry {
    /// Declare actual site positions and chemical identities in the body frame.
    pub fn new(sites: Vec<[f64; 3]>, species: Vec<u32>) -> Result<Self, &'static str> {
        if sites.is_empty() || sites.len() != species.len() || species.contains(&0)
            || sites.iter().flatten().any(|value| !value.is_finite()) {
            return Err("invalid rigid-body site geometry");
        }
        Ok(Self { sites, species })
    }

    /// Expand centers followed by exponential-map rotations into Cartesian sites.
    pub fn expand(&self, molecules: usize, state: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(state.len(), 6 * molecules, "rigid state has a center and rotation per molecule");
        let mut points = Vec::with_capacity(3 * molecules * self.sites.len());
        for molecule in 0..molecules {
            let offset = 3 * molecules + 3 * molecule;
            let rotation = [state[offset], state[offset + 1], state[offset + 2]];
            let theta = rotation.iter().map(|value| value * value).sum::<f64>().sqrt();
            for site in &self.sites {
                let rotated = if theta == 0.0 {
                    *site
                } else {
                    let axis = rotation.map(|value| value / theta);
                    let projection = axis.iter().zip(site).map(|(a, b)| a * b).sum::<f64>();
                    let cross = [axis[1] * site[2] - axis[2] * site[1], axis[2] * site[0] - axis[0] * site[2], axis[0] * site[1] - axis[1] * site[0]];
                    let (sin, cos) = theta.sin_cos();
                    std::array::from_fn(|k| cos * site[k] + sin * cross[k] + (1.0 - cos) * projection * axis[k])
                };
                for k in 0..3 { points.push(state[3 * molecule + k] + rotated[k]); }
            }
        }
        Array1::from(points)
    }

    /// Sorted site distances in stable species-pair blocks.
    pub fn describe(&self, molecules: usize, state: ArrayView1<f64>) -> Array1<f64> {
        let points = self.expand(molecules, state);
        let n = molecules * self.sites.len();
        let mut blocks = BTreeMap::<(u32, u32), Vec<f64>>::new();
        for i in 0..n {
            for j in i + 1..n {
                let a = self.species[i % self.sites.len()];
                let b = self.species[j % self.sites.len()];
                let distance = (0..3).map(|k| (points[3 * i + k] - points[3 * j + k]).powi(2)).sum::<f64>().sqrt();
                blocks.entry((a.min(b), a.max(b))).or_default().push(distance);
            }
        }
        for distances in blocks.values_mut() { distances.sort_by(f64::total_cmp); }
        Array1::from(blocks.into_values().flatten().collect::<Vec<_>>())
    }
}
