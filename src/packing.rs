//! One surface for packing mutates.
//!
//! Close-packed morphologies (fcc, hcp, Marks, Mackay) differ by stacking.
//! The operations that change stacking without a CMA-style rebuild are
//! here: twin a dense plane as a proposal, or close the core's orbits
//! after a newly entered basin. Cut-and-splice is not this surface.
//! Catalog Leave is not this surface.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// Twin across one of the densest planes. Always a coordinate vector;
/// identity if no plane is found.
pub fn propose_twin<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    n_points: usize,
    rng: &mut R,
) -> Array1<f64> {
    crate::twin::propose(x, n_points, rng)
}

/// Close surface atoms onto the core point-group orbits. `None` if the
/// core has no group that moves anyone.
pub fn on_new_basin(
    x: ArrayView1<f64>,
    n_points: usize,
    tolerance: f64,
    merge_radius: f64,
    core_fraction: f64,
    min_separation: f64,
) -> Option<Array1<f64>> {
    crate::symmetrise::orbit_complete_core(
        x,
        n_points,
        tolerance,
        merge_radius,
        core_fraction,
        min_separation,
    )
}

/// Which packing mutates a hop config has armed. Flags stay on [`crate::methods::cluster_hopping::Config`]; this is the one place that names them together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackingSurface {
    /// [`crate::methods::cluster_hopping::MoveLibrary::Twin`] is in the library.
    pub twin_as_move: bool,
    /// Orbit completion after a newly accepted basin.
    pub orbit_on_new: bool,
    /// Wales--Doye angular relocation of the worst-bound atom.
    pub angular: bool,
    /// Point-group symmetrisation after a newly accepted basin.
    pub psym_on_new: bool,
}

impl PackingSurface {
    /// Assemble from the hop flags. No new CLI.
    pub fn from_hop_flags(
        twin_as_move: bool,
        orbit_on_new: bool,
        angular: bool,
        psym_on_new: bool,
    ) -> Self {
        Self {
            twin_as_move,
            orbit_on_new,
            angular,
            psym_on_new,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn twin_propose_returns_a_state_of_the_same_length() {
        let x = array![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.87, 0.0, 0.5, 0.29, 0.82
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let y = propose_twin(x.view(), 4, &mut rng);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn on_new_basin_is_optional_on_a_tiny_cluster() {
        let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(on_new_basin(x.view(), 3, 0.2, 0.3, 0.6, 0.4).is_none());
    }
}
