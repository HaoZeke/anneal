//! Gupta potential in the Cleri–Rosato form.
//!
//! The total energy is
//!
//! ```text
//! E = Σ_{i<j} 2 A_{ij} exp(−p_{ij} (r_{ij}/r0_{ij} − 1))
//!   − Σ_i √(Σ_{j≠i} ξ_{ij}² exp(−2 q_{ij} (r_{ij}/r0_{ij} − 1)))
//! ```
//!
//! which is the pair-plus-embedding form Doye writes for the Cambridge
//! zinc and cadmium clusters. A pure metal is the one-species case.
//! Copper–gold uses one parameter triple for each of Cu–Cu, Au–Au and
//! Cu–Au. There is no cutoff: the Cambridge minima reproduced here are
//! the all-pairs sums.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

/// One ordered pair of species in a [`Gupta`] cluster.
#[derive(Debug, Clone, Copy)]
struct GuptaPair {
    a: f64,
    xi: f64,
    p: f64,
    q: f64,
    r0: f64,
}

/// Cleri–Rosato zinc, from Doye, Phys. Rev. B 68, 195418 (2003), Table 1.
///
/// Lengths are in units of `r0`, which is how the Cambridge point files
/// are written. `A = 0.1477` eV, `ξ = 0.8900` eV, `p = 9.689`, `q = 4.602`.
const ZINC: GuptaPair = GuptaPair {
    a: 0.1477,
    xi: 0.8900,
    p: 9.689,
    q: 4.602,
    r0: 1.0,
};

/// Cleri–Rosato copper. `A`, `ξ`, `p`, `q`, `r0` in eV and Å.
const COPPER: GuptaPair = GuptaPair {
    a: 0.0855,
    xi: 1.224,
    p: 10.960,
    q: 2.278,
    r0: 2.556,
};

/// Cleri–Rosato gold.
const GOLD: GuptaPair = GuptaPair {
    a: 0.2061,
    xi: 1.790,
    p: 10.229,
    q: 4.036,
    r0: 2.884,
};

/// Cleri–Rosato copper–gold cross term.
///
/// `r0` is the copper nearest-neighbour distance. This is the cross term
/// that reproduces every composition in the Cambridge `CuAu_38` archive.
const COPPER_GOLD: GuptaPair = GuptaPair {
    a: 0.1539,
    xi: 1.5605,
    p: 11.05,
    q: 3.0475,
    r0: 2.556,
};

/// Atomic number used for copper in the species vector.
pub const COPPER_Z: u32 = 29;

/// Atomic number used for gold in the species vector.
pub const GOLD_Z: u32 = 79;

/// A Gupta cluster with a fixed composition.
///
/// Atom `i` keeps `species[i]` for the whole search. A homotop exchanges
/// coordinates and leaves this vector where it is.
#[derive(Debug, Clone)]
pub struct Gupta {
    n_points: usize,
    /// Species slot of each atom, in `0..n_species`.
    slot: Vec<u8>,
    n_species: usize,
    /// Row-major `n_species * n_species` pair table.
    pairs: Vec<GuptaPair>,
    bounds: Bounds<f64>,
}

impl Gupta {
    /// Zinc cluster of `n_points` atoms, lengths in units of `r0`.
    pub fn zinc(n_points: usize) -> Self {
        Self::from_slots(vec![0; n_points], 1, vec![ZINC])
    }

    /// Copper atoms followed by gold atoms.
    ///
    /// The species labels are [`COPPER_Z`] then [`GOLD_Z`]. A homotop keeps
    /// that labelling and exchanges coordinates.
    pub fn copper_gold(n_copper: usize, n_gold: usize) -> Self {
        Self::elements(&Self::copper_gold_species(n_copper, n_gold))
    }

    /// Copper–gold cluster whose atomic numbers are `labels`, one per atom.
    ///
    /// Cambridge `CuAu_38` files interleave the two species. Type `0` in
    /// those files is copper and type `1` is gold; this method takes the
    /// atomic numbers.
    pub fn elements(labels: &[u32]) -> Self {
        assert!(
            labels
                .iter()
                .all(|label| *label == COPPER_Z || *label == GOLD_Z),
            "copper–gold labels are Cu and Au"
        );
        let slot = labels
            .iter()
            .map(|label| u8::from(*label == GOLD_Z))
            .collect();
        Self::from_slots(slot, 2, vec![COPPER, COPPER_GOLD, COPPER_GOLD, GOLD])
    }

    /// Species labels parallel to the atoms, copper then gold.
    pub fn copper_gold_species(n_copper: usize, n_gold: usize) -> Vec<u32> {
        let mut species = vec![COPPER_Z; n_copper];
        species.extend(std::iter::repeat_n(GOLD_Z, n_gold));
        species
    }

    fn from_slots(slot: Vec<u8>, n_species: usize, pairs: Vec<GuptaPair>) -> Self {
        let n_points = slot.len();
        assert!(n_points >= 2, "a Gupta cluster needs at least two atoms");
        assert_eq!(pairs.len(), n_species * n_species, "pair table is square");
        assert!(
            slot.iter().all(|index| (*index as usize) < n_species),
            "species slot out of range"
        );
        let r0 = pairs.iter().fold(0.0_f64, |best, pair| best.max(pair.r0));
        let extent = 4.0 * r0 * (n_points as f64).cbrt();
        Self {
            n_points,
            slot,
            n_species,
            pairs,
            bounds: Bounds::new(
                Array1::from_elem(3 * n_points, -extent),
                Array1::from_elem(3 * n_points, extent),
                0.0,
            ),
        }
    }

    /// Atoms in the cluster.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    fn pair(&self, i: usize, j: usize) -> GuptaPair {
        let row = self.slot[i] as usize;
        let column = self.slot[j] as usize;
        self.pairs[row * self.n_species + column]
    }

    /// Energy and Cartesian gradient in one pair pass.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_points;
        assert_eq!(x.len(), 3 * n, "Gupta state is 3 coordinates per atom");
        let coordinates = x.as_slice().expect("Gupta state is contiguous");
        let mut energy = 0.0;
        let mut rho = vec![0.0; n];
        let mut stored = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            let base_i = 3 * i;
            for j in (i + 1)..n {
                let base_j = 3 * j;
                let d = [
                    coordinates[base_i] - coordinates[base_j],
                    coordinates[base_i + 1] - coordinates[base_j + 1],
                    coordinates[base_i + 2] - coordinates[base_j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= 0.0 {
                    return (f64::INFINITY, Array1::zeros(x.len()));
                }
                let pair = self.pair(i, j);
                let r = r2.sqrt();
                let u = r / pair.r0 - 1.0;
                let phi = 2.0 * pair.a * (-pair.p * u).exp();
                let psi = pair.xi * pair.xi * (-2.0 * pair.q * u).exp();
                energy += phi;
                rho[i] += psi;
                rho[j] += psi;
                let dphi_dr = -(pair.p / pair.r0) * phi;
                let dpsi_dr = -(2.0 * pair.q / pair.r0) * psi;
                stored.push((i, j, d, r, dphi_dr, dpsi_dr));
            }
        }
        for density in &rho {
            energy -= density.sqrt();
        }
        let mut gradient = vec![0.0; 3 * n];
        for (i, j, d, r, dphi_dr, dpsi_dr) in stored {
            let embed = -0.5 * (rho[i].sqrt().recip() + rho[j].sqrt().recip());
            let coef = (dphi_dr + embed * dpsi_dr) / r;
            for k in 0..3 {
                gradient[3 * i + k] += coef * d[k];
                gradient[3 * j + k] -= coef * d[k];
            }
        }
        (energy, Array1::from_vec(gradient))
    }
}

impl Objective<f64> for Gupta {
    fn dim(&self) -> usize {
        3 * self.n_points
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Gupta {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        3 * self.n_points
    }
}

impl DifferentiableObjective<f64> for Gupta {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        Gupta::value_and_gradient(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinates(text: &str) -> Array1<f64> {
        let mut values = Vec::new();
        for line in text.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() == 3
                && let Ok(row) = fields
                    .iter()
                    .map(|word| word.parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
            {
                values.extend(row);
            }
        }
        Array1::from_vec(values)
    }

    /// Cambridge `CuAu_38` row: species `0` is copper and `1` is gold.
    fn copper_gold_file(text: &str) -> (Gupta, Array1<f64>) {
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for line in text.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 4 {
                continue;
            }
            let Ok(kind) = fields[0].parse::<u32>() else {
                continue;
            };
            let Ok(row) = fields[1..]
                .iter()
                .map(|word| word.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
            else {
                continue;
            };
            labels.push(if kind == 0 { COPPER_Z } else { GOLD_Z });
            values.extend(row);
        }
        (Gupta::elements(&labels), Array1::from_vec(values))
    }

    #[test]
    fn a_zinc_dimer_at_r0_is_the_closed_form() {
        let pot = Gupta::zinc(2);
        let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let energy = pot.eval(x.view());
        let closed = 2.0 * ZINC.a - 2.0 * ZINC.xi;
        assert!(
            (energy - closed).abs() < 1e-12,
            "dimer {energy} against {closed}"
        );
    }

    #[test]
    fn the_gradient_matches_a_central_difference() {
        let pot = Gupta::zinc(4);
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.05, 0.02, 0.0, 0.4, 0.9, -0.1, 0.2, 0.3, 1.1,
        ]);
        let (_energy, gradient) = pot.value_and_gradient(x.view());
        let step = 1.0e-5;
        for index in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[index] += step;
            minus[index] -= step;
            let slope = (pot.eval(plus.view()) - pot.eval(minus.view())) / (2.0 * step);
            assert!(
                (gradient[index] - slope).abs() < 1e-7,
                "component {index}: analytic {} central {slope}",
                gradient[index]
            );
        }
    }

    #[test]
    fn trait_methods_match_the_fused_evaluation() {
        let pot = Gupta::copper_gold(2, 2);
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 2.6, 0.1, 0.0, 1.2, 2.3, 0.2, 1.3, 0.4, 2.5,
        ]);
        let (energy, gradient) = pot.value_and_gradient(x.view());
        assert_eq!(energy, pot.eval(x.view()));
        assert_eq!(gradient, pot.grad(x.view()));
        assert_eq!(Objective::dim(&pot), 12);
    }

    fn holds(pot: &Gupta, text: &str, listed: f64) {
        let x = coordinates(text);
        assert_eq!(x.len(), 3 * pot.n_points());
        let energy = pot.eval(x.view());
        assert!(
            (energy - listed).abs() < 1e-6,
            "energy {energy:.10} is not the Cambridge {listed:.8}"
        );
    }

    #[test]
    fn zinc_reproduces_the_cambridge_minima() {
        // Doye, Phys. Rev. B 68, 195418 (2003), as listed on the Cambridge
        // Cluster Database Gupta zinc table. The point files are in units of r0.
        let cases = [
            (
                13_usize,
                include_str!("../../tests/fixtures/ccd/gupta_zn/13"),
                -17.032142,
            ),
            (
                38,
                include_str!("../../tests/fixtures/ccd/gupta_zn/38"),
                -50.499983,
            ),
            (
                55,
                include_str!("../../tests/fixtures/ccd/gupta_zn/55"),
                -73.310032,
            ),
            (
                75,
                include_str!("../../tests/fixtures/ccd/gupta_zn/75"),
                -100.203580,
            ),
            (
                76,
                include_str!("../../tests/fixtures/ccd/gupta_zn/76"),
                -101.545341,
            ),
            (
                77,
                include_str!("../../tests/fixtures/ccd/gupta_zn/77"),
                -102.900278,
            ),
            (
                78,
                include_str!("../../tests/fixtures/ccd/gupta_zn/78"),
                -104.238674,
            ),
            (
                79,
                include_str!("../../tests/fixtures/ccd/gupta_zn/79"),
                -105.595357,
            ),
            (
                80,
                include_str!("../../tests/fixtures/ccd/gupta_zn/80"),
                -106.927940,
            ),
        ];
        for (n, text, listed) in cases {
            holds(&Gupta::zinc(n), text, listed);
        }
    }

    #[test]
    fn copper_gold_reproduces_the_cambridge_archive() {
        // Cambridge Cluster Database CuAu_38, energies from the file headers.
        // The same Cleri–Rosato table hits every composition in that archive
        // to better than 1e-8; these five lock the pure limits and a mix.
        let cases = [
            (
                0_usize,
                38_usize,
                include_str!("../../tests/fixtures/ccd/cuau38/Cu0Au38.txt"),
                -130.54001708,
            ),
            (
                38,
                0,
                include_str!("../../tests/fixtures/ccd/cuau38/Cu38Au0.txt"),
                -108.96759055,
            ),
            (
                6,
                32,
                include_str!("../../tests/fixtures/ccd/cuau38/Cu6Au32.txt"),
                -130.79994836,
            ),
            (
                19,
                19,
                include_str!("../../tests/fixtures/ccd/cuau38/Cu19Au19.txt"),
                -122.99148391,
            ),
            (
                34,
                4,
                include_str!("../../tests/fixtures/ccd/cuau38/Cu34Au4.txt"),
                -112.21399705,
            ),
        ];
        for (_n_copper, _n_gold, text, listed) in cases {
            let (pot, x) = copper_gold_file(text);
            let energy = pot.eval(x.view());
            assert!(
                (energy - listed).abs() < 1e-6,
                "energy {energy:.10} is not the Cambridge {listed:.8}"
            );
        }
    }
}
