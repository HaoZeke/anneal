//! Invariant descriptors of rigid bodies represented by centers and rotations.
//!
//! The molecular geometry projects generalized coordinates onto actual sites.
//! Species-separated distance spectra retain orientation information without
//! treating equivalent atoms or equivalent rotation charts as distinct states.

use ndarray::{Array1, ArrayView1};

/// Site geometry relative to the center used by the objective's rigid-body chart.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RigidBodyGeometry {
    sites: Vec<[f64; 3]>,
    species: Vec<u32>,
    #[serde(skip)]
    block_keys: Vec<(u32, u32)>,
    #[serde(skip)]
    block_of_pair: Vec<usize>,
}

impl RigidBodyGeometry {
    /// Declare actual site positions and chemical identities in the body frame.
    pub fn new(sites: Vec<[f64; 3]>, species: Vec<u32>) -> Result<Self, &'static str> {
        if sites.is_empty()
            || sites.len() != species.len()
            || species.contains(&0)
            || sites.iter().flatten().any(|value| !value.is_finite())
        {
            return Err("invalid rigid-body site geometry");
        }
        let mut block_keys = Vec::new();
        for &a in &species {
            for &b in &species {
                block_keys.push((a.min(b), a.max(b)));
            }
        }
        block_keys.sort_unstable();
        block_keys.dedup();
        let block_of_pair = species
            .iter()
            .flat_map(|&a| {
                let keys = &block_keys;
                species.iter().map(move |&b| {
                    keys.binary_search(&(a.min(b), a.max(b)))
                        .expect("species pair was collected")
                })
            })
            .collect();
        Ok(Self {
            sites,
            species,
            block_keys,
            block_of_pair,
        })
    }

    /// Expand centers followed by exponential-map rotations into Cartesian sites.
    pub fn expand(&self, molecules: usize, state: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(
            state.len(),
            6 * molecules,
            "rigid state has a center and rotation per molecule"
        );
        let mut points = Vec::with_capacity(3 * molecules * self.sites.len());
        for molecule in 0..molecules {
            let offset = 3 * molecules + 3 * molecule;
            let rotation = [state[offset], state[offset + 1], state[offset + 2]];
            let theta = rotation
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            for site in &self.sites {
                let rotated = if theta == 0.0 {
                    *site
                } else {
                    let axis = rotation.map(|value| value / theta);
                    let projection = axis.iter().zip(site).map(|(a, b)| a * b).sum::<f64>();
                    let cross = [
                        axis[1] * site[2] - axis[2] * site[1],
                        axis[2] * site[0] - axis[0] * site[2],
                        axis[0] * site[1] - axis[1] * site[0],
                    ];
                    let (sin, cos) = theta.sin_cos();
                    std::array::from_fn(|k| {
                        cos * site[k] + sin * cross[k] + (1.0 - cos) * projection * axis[k]
                    })
                };
                for k in 0..3 {
                    points.push(state[3 * molecule + k] + rotated[k]);
                }
            }
        }
        Array1::from(points)
    }

    /// Body-frame site positions, in declaration order.
    pub fn sites(&self) -> &[[f64; 3]] {
        &self.sites
    }

    /// Atomic number of each site, in declaration order.
    pub fn species(&self) -> &[u32] {
        &self.species
    }

    /// Largest body-frame site distance from the centre.
    ///
    /// A rigid chart uses this reach to size the container: the preset's
    /// reduced-unit sphere is the monoatomic cluster radius, and a water
    /// with sites an angstrom from the centre does not fit in it.
    pub fn site_reach(&self) -> f64 {
        self.sites
            .iter()
            .map(|site| (site[0] * site[0] + site[1] * site[1] + site[2] * site[2]).sqrt())
            .fold(0.0_f64, f64::max)
    }

    /// Sorted site distances in stable species-pair blocks.
    pub fn describe(&self, molecules: usize, state: ArrayView1<f64>) -> Array1<f64> {
        let points = self.expand(molecules, state);
        let sites_per_molecule = self.sites.len();
        let n = molecules * sites_per_molecule;
        let mut blocks: Vec<Vec<f64>> = self
            .block_keys
            .iter()
            .map(|&(a, b)| {
                let count_a =
                    molecules * self.species.iter().filter(|&&species| species == a).count();
                let count_b =
                    molecules * self.species.iter().filter(|&&species| species == b).count();
                let capacity = if a == b {
                    count_a * count_a.saturating_sub(1) / 2
                } else {
                    count_a * count_b
                };
                Vec::with_capacity(capacity)
            })
            .collect();
        for i in 0..n {
            for j in i + 1..n {
                let distance = (0..3)
                    .map(|k| (points[3 * i + k] - points[3 * j + k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let site_i = i % sites_per_molecule;
                let site_j = j % sites_per_molecule;
                blocks[self.block_of_pair[site_i * sites_per_molecule + site_j]].push(distance);
            }
        }
        for distances in &mut blocks {
            distances.sort_unstable_by(f64::total_cmp);
        }
        Array1::from(blocks.into_iter().flatten().collect::<Vec<_>>())
    }
}
