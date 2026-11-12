//! Invariant descriptors of rigid bodies represented by centers and rotations.
//!
//! The molecular geometry projects generalized coordinates onto actual sites.
//! Species-separated distance spectra retain orientation information without
//! treating equivalent atoms or equivalent rotation charts as distinct states.

use std::cell::RefCell;

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
    ///
    /// The block buffers are reused. A hop asks for this spectrum several
    /// times, and building a fresh vector of vectors on each ask was a
    /// visible part of the time spent outside the potential.
    pub fn describe(&self, molecules: usize, state: ArrayView1<f64>) -> Array1<f64> {
        // A hop asks for the spectrum of one minimum several times. The last
        // answer is reused when the centres and rotations are unchanged.
        thread_local! {
            static LAST: RefCell<Option<(usize, usize, Vec<f64>, Vec<f64>)>> = const { RefCell::new(None) };
        }
        let marker = self as *const Self as usize;
        let state_key = state
            .as_slice()
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|| state.iter().copied().collect());
        if let Some(hit) = LAST.with(|cell| {
            cell.borrow().as_ref().and_then(|cached| {
                (cached.0 == marker && cached.1 == molecules && cached.2 == state_key)
                    .then(|| cached.3.clone())
            })
        }) {
            return Array1::from(hit);
        }
        let points = self.expand(molecules, state);
        let sites_per_molecule = self.sites.len();
        let n = molecules * sites_per_molecule;
        let nblocks = self.block_keys.len();
        thread_local! {
            static BLOCKS: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::new());
        }
        BLOCKS.with(|cell| {
            let mut blocks = cell.borrow_mut();
            if blocks.len() < nblocks {
                blocks.resize_with(nblocks, Vec::new);
            }
            for block in blocks.iter_mut().take(nblocks) {
                block.clear();
            }
            for i in 0..n {
                let pi = 3 * i;
                for j in i + 1..n {
                    let pj = 3 * j;
                    let mut distance = 0.0;
                    for k in 0..3 {
                        let delta = points[pi + k] - points[pj + k];
                        distance += delta * delta;
                    }
                    let site_i = i % sites_per_molecule;
                    let site_j = j % sites_per_molecule;
                    blocks[self.block_of_pair[site_i * sites_per_molecule + site_j]]
                        .push(distance.sqrt());
                }
            }
            for block in blocks.iter_mut().take(nblocks) {
                block.sort_unstable_by(f64::total_cmp);
            }
            let total = blocks.iter().take(nblocks).map(|block| block.len()).sum();
            let mut flat = Vec::with_capacity(total);
            for block in blocks.iter().take(nblocks) {
                flat.extend_from_slice(block);
            }
            LAST.with(|cell| {
                *cell.borrow_mut() = Some((marker, molecules, state_key, flat.clone()));
            });
            Array1::from(flat)
        })
    }
}
