//! Bond-graph preconditioning for molecular quenches.
//!
//! A covalent molecule's Hessian spans stiff bond stretches and soft
//! torsions, two or three decades apart. A Cartesian quasi-Newton descent
//! spends most of its evaluations learning that spread. Measured on a
//! 42-atom peptide at GFN2-xTB, the crate's L-BFGS took 233 calls to settle
//! a structure that xtb's own optimiser settles in 44 cycles.
//!
//! This is the exponential preconditioner of Packwood and coworkers
//! (J. Chem. Phys. 144, 164109, 2016). It is a graph Laplacian over atom
//! pairs, weighted by how close each pair is to its covalent bond length:
//!
//! \[ w_{ij} = \exp\left(-A\left(\frac{r_{ij}}{r^0_{ij}} - 1\right)\right),
//!    \quad r_{ij} < c\, r^0_{ij}, \]
//!
//! with \(r^0_{ij}\) the sum of Cordero covalent radii, \(A = 3\) and
//! \(c = 2\). Bonded atoms are coupled at weight about one and distant
//! atoms not at all, so a preconditioned step moves bonded atoms together.
//! A small diagonal term keeps rigid translation finite. The matrix acts
//! on each Cartesian component alike, so its factor is the size of the
//! atom count, not three times it.
//!
//! The descent runs in reduced coordinates \(z\), with
//! \(x = x_0 + L^{-\top} z\) and \(P = L L^\top\). There
//! \(\nabla_z E = L^{-1} \nabla_x E\), and plain L-BFGS on \(z\) is L-BFGS
//! on \(x\) whose initial inverse Hessian is \(P^{-1}\). Energies are
//! unchanged, so a caller's convergence test stays on the Cartesian
//! gradient.

use ndarray::{Array1, ArrayView1};

use crate::methods::cluster_hopping::covalent_radius;

/// Decay of a pair's weight past its bond length, Packwood's `A`.
const DECAY: f64 = 3.0;
/// Pairs farther apart than this multiple of their covalent sum are uncoupled.
const CUTOFF: f64 = 2.0;
/// Each atom's own diagonal weight, a tenth of one bond.
const STABILISER: f64 = 0.1;

/// Cholesky factor of a bond-graph preconditioner at one geometry.
#[derive(Debug, Clone)]
pub struct BondPreconditioner {
    atoms: usize,
    /// Lower-triangular factor, row major.
    lower: Vec<f64>,
}

impl BondPreconditioner {
    /// The preconditioner at `x`, or `None` when the coordinates do not
    /// match the species or no species has a covalent radius.
    pub fn new(x: ArrayView1<f64>, species: &[u32]) -> Option<Self> {
        let atoms = species.len();
        if atoms == 0 || x.len() != 3 * atoms {
            return None;
        }
        let radii: Vec<f64> = species.iter().map(|&z| covalent_radius(z)).collect();
        if radii.iter().all(|&r| r <= 0.0) {
            return None;
        }
        let mut p = vec![0.0; atoms * atoms];
        for i in 0..atoms {
            p[i * atoms + i] = STABILISER;
        }
        for i in 0..atoms {
            for j in i + 1..atoms {
                let bond = radii[i] + radii[j];
                if radii[i] <= 0.0 || radii[j] <= 0.0 {
                    continue;
                }
                let r = (0..3)
                    .map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if r >= CUTOFF * bond {
                    continue;
                }
                let w = (-DECAY * (r / bond - 1.0)).exp();
                p[i * atoms + j] -= w;
                p[j * atoms + i] -= w;
                p[i * atoms + i] += w;
                p[j * atoms + j] += w;
            }
        }
        cholesky(&mut p, atoms).then_some(Self { atoms, lower: p })
    }

    /// The reduced gradient \(L^{-1} g\).
    pub fn reduce(&self, gradient: ArrayView1<f64>) -> Array1<f64> {
        let n = self.atoms;
        let mut out = gradient.to_owned();
        for k in 0..3 {
            for i in 0..n {
                let mut v = out[3 * i + k];
                for j in 0..i {
                    v -= self.lower[i * n + j] * out[3 * j + k];
                }
                out[3 * i + k] = v / self.lower[i * n + i];
            }
        }
        out
    }

    /// The Cartesian point \(x_0 + L^{-\top} z\).
    pub fn expand(&self, origin: ArrayView1<f64>, reduced: ArrayView1<f64>) -> Array1<f64> {
        let n = self.atoms;
        let mut step = reduced.to_owned();
        for k in 0..3 {
            for i in (0..n).rev() {
                let mut v = step[3 * i + k];
                for j in i + 1..n {
                    v -= self.lower[j * n + i] * step[3 * j + k];
                }
                step[3 * i + k] = v / self.lower[i * n + i];
            }
        }
        step + origin
    }
}

/// In-place Cholesky factorisation of a symmetric `n` by `n` matrix,
/// leaving the lower factor and zeroing the upper triangle. False when the
/// matrix is not positive definite.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
        for i in 0..j {
            a[i * n + j] = 0.0;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ethane, staggered, in angstrom.
    fn ethane() -> (Vec<u32>, Array1<f64>) {
        let species = vec![6, 6, 1, 1, 1, 1, 1, 1];
        let x = Array1::from(vec![
            0.0, 0.0, 0.765, 0.0, 0.0, -0.765, 1.018, 0.0, 1.164, -0.509, 0.882, 1.164, -0.509,
            -0.882, 1.164, -1.018, 0.0, -1.164, 0.509, -0.882, -1.164, 0.509, 0.882, -1.164,
        ]);
        (species, x)
    }

    fn matrix(x: ArrayView1<f64>, species: &[u32]) -> Vec<f64> {
        let n = species.len();
        let p = BondPreconditioner::new(x, species).expect("preconditioner");
        let mut full = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                full[i * n + j] = (0..n)
                    .map(|k| p.lower[i * n + k] * p.lower[j * n + k])
                    .sum();
            }
        }
        full
    }

    #[test]
    fn a_bonded_pair_couples_and_a_distant_one_does_not() {
        let (species, x) = ethane();
        let p = matrix(x.view(), &species);
        let n = species.len();
        assert!(p[1] < -0.5, "carbon pair weight {}", -p[1]);
        let mut far = x.clone();
        for v in far.iter_mut().skip(3).take(3) {
            *v += 50.0;
        }
        let q = matrix(far.view(), &species);
        assert_eq!(q[1], 0.0);
        assert!((0..n).all(|i| p[i * n + i] > 0.0));
    }

    /// One unit step along the reduced gradient of the preconditioner's own
    /// quadratic lands on its minimum: the reduced Hessian is the identity.
    #[test]
    fn the_reduced_hessian_of_its_own_quadratic_is_the_identity() {
        let (species, x) = ethane();
        let n = species.len();
        let p = matrix(x.view(), &species);
        let pre = BondPreconditioner::new(x.view(), &species).unwrap();
        let centre = x.mapv(|v| 0.9 * v + 0.05);
        let mut g = Array1::zeros(3 * n);
        for k in 0..3 {
            for i in 0..n {
                g[3 * i + k] = (0..n)
                    .map(|j| p[i * n + j] * (x[3 * j + k] - centre[3 * j + k]))
                    .sum();
            }
        }
        let landed = pre.expand(x.view(), (-pre.reduce(g.view())).view());
        for (a, b) in landed.iter().zip(centre.iter()) {
            assert!((a - b).abs() < 1e-10, "{a} against {b}");
        }
    }

    #[test]
    fn a_mismatched_geometry_has_no_preconditioner() {
        let (species, x) = ethane();
        assert!(BondPreconditioner::new(x.slice(ndarray::s![..9]), &species).is_none());
        assert!(BondPreconditioner::new(x.view(), &[0; 8]).is_none());
    }
}
