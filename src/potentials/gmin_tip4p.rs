//! Rigid TIP4P on the GMIN angle-axis parameter set, stored in the block chart.
//!
//! Constants are `DEFTIP4` in GMIN `source/newtip.f90`: C6 is 2552.24
//! kJ/mol Å^6, C12 is 2510.4e3 kJ/mol Å^12, and the Coulomb prefactor is
//! 1389.354848. Charges are 0, 0.52, 0.52, -1.04. Geometry is rOH 0.9572 Å,
//! angle 104.52 degrees, M site 0.15 Å from oxygen along the bisector.
//! Masses are 16, 1, 1, 0.
//!
//! GMIN stores each molecule as centre then angle-axis, and its rotation is
//! the opposite sense of the Rodrigues map this hop already uses for rigid
//! bodies. The objective here keeps centres in one block and Rodrigues
//! rotations in the next, so recentring leaves orientations alone. Conversion
//! at the boundary negates the angle-axis vectors.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

use super::tip4p::{cross, left_jacobian_t, mat_vec, rotation_matrix, rotation_vector};

/// Lennard-Jones attractive coefficient, kJ/mol Å^6.
pub const C6: f64 = 2552.24;
/// Lennard-Jones repulsive coefficient, kJ/mol Å^12.
pub const C12: f64 = 2510.4e3;
/// Coulomb prefactor, kJ/mol Å e^-2.
pub const COULOMB: f64 = 1389.354848;
/// O-H bond length, angstrom.
pub const R_OH: f64 = 0.9572;
/// H-O-H angle, degrees.
pub const HOH_DEG: f64 = 104.52;
/// M-site distance from oxygen, angstrom.
pub const R_OM: f64 = 0.15;
const Q_H: f64 = 0.52;
const Q_M: f64 = -1.04;
const MASS_O: f64 = 16.0;
const MASS_H: f64 = 1.0;

/// Rigid TIP4P cluster in the block Rodrigues chart.
#[derive(Debug, Clone)]
pub struct GminTip4p {
    n_molecules: usize,
    body: [[f64; 3]; 4],
    bounds: Bounds<f64>,
}

impl GminTip4p {
    /// A cluster of `n_molecules` rigid waters.
    pub fn new(n_molecules: usize) -> Self {
        assert!(n_molecules >= 1, "TIP4P needs at least one molecule");
        let dim = 6 * n_molecules;
        let extent = 8.0 * (n_molecules as f64).cbrt().max(1.0);
        Self {
            n_molecules,
            body: reference_sites(),
            bounds: Bounds::new(
                Array1::from_elem(dim, -extent),
                Array1::from_elem(dim, extent),
                0.0,
            ),
        }
    }

    /// Molecules in the cluster.
    pub fn n_molecules(&self) -> usize {
        self.n_molecules
    }

    /// Energy and rigid-body gradient. Centres occupy the first `3 n` entries.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_molecules;
        assert_eq!(x.len(), 6 * n, "TIP4P state is 6 coordinates per molecule");
        let mut sites = vec![[[0.0; 3]; 4]; n];
        let mut rotations = vec![[[0.0; 3]; 3]; n];
        for i in 0..n {
            let centre = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
            let omega = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
            rotations[i] = rotation_matrix(omega);
            for (site, body) in sites[i].iter_mut().zip(self.body.iter()) {
                let placed = mat_vec(rotations[i], *body);
                *site = [
                    centre[0] + placed[0],
                    centre[1] + placed[1],
                    centre[2] + placed[2],
                ];
            }
        }
        let (energy, force) = site_energy_forces(&sites);
        let mut gradient = Array1::zeros(6 * n);
        for i in 0..n {
            let mut torque = [0.0; 3];
            for a in 0..4 {
                for k in 0..3 {
                    gradient[3 * i + k] += force[i][a][k];
                }
                let relative = mat_vec(rotations[i], self.body[a]);
                let arm = cross(relative, force[i][a]);
                for k in 0..3 {
                    torque[k] += arm[k];
                }
            }
            let omega = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
            let pulled = mat_vec(left_jacobian_t(omega), torque);
            for k in 0..3 {
                gradient[3 * n + 3 * i + k] = pulled[k];
            }
        }
        (energy, gradient)
    }
}

impl Objective<f64> for GminTip4p {
    fn dim(&self) -> usize {
        6 * self.n_molecules
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for GminTip4p {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        6 * self.n_molecules
    }
}

impl DifferentiableObjective<f64> for GminTip4p {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        GminTip4p::value_and_gradient(self, x)
    }
}

/// O, H, H, M in the centre-of-mass frame. Masses are 16, 1, 1, 0.
pub fn reference_sites() -> [[f64; 3]; 4] {
    let half = (HOH_DEG * 0.5).to_radians();
    let (sin_half, cos_half) = half.sin_cos();
    let oxygen = [0.0, 0.0, 0.0];
    let hydrogen_1 = [0.0, sin_half * R_OH, cos_half * R_OH];
    let hydrogen_2 = [0.0, -sin_half * R_OH, cos_half * R_OH];
    let m_site = [0.0, 0.0, R_OM];
    let sites = [oxygen, hydrogen_1, hydrogen_2, m_site];
    let mass = MASS_O + 2.0 * MASS_H;
    let mut centre = [0.0; 3];
    let masses = [MASS_O, MASS_H, MASS_H, 0.0];
    for (site, mass_i) in sites.iter().zip(masses.iter()) {
        for k in 0..3 {
            centre[k] += mass_i * site[k];
        }
    }
    for value in centre.iter_mut() {
        *value /= mass;
    }
    sites.map(|site| {
        [
            site[0] - centre[0],
            site[1] - centre[1],
            site[2] - centre[2],
        ]
    })
}

/// O, H, H geometry used to key rigid basins. The M site is not an atom.
pub fn gmin_water_geometry() -> crate::rigid_body::RigidBodyGeometry {
    let sites = reference_sites();
    crate::rigid_body::RigidBodyGeometry::new(sites[..3].to_vec(), vec![8, 1, 1])
        .expect("water O-H-H geometry is finite")
}

/// Block Rodrigues coordinates from Cartesian O, H, H triples.
pub fn rigid_from_ohh(cartesian: &[f64]) -> Array1<f64> {
    assert!(
        cartesian.len().is_multiple_of(9) && !cartesian.is_empty(),
        "OHH coordinates are O, H, H repeated"
    );
    let n = cartesian.len() / 9;
    let references = reference_sites();
    let body = [references[0], references[1], references[2]];
    let mut state = Array1::zeros(6 * n);
    for molecule in 0..n {
        let base = 9 * molecule;
        let mut lab = [[0.0; 3]; 3];
        for atom in 0..3 {
            for k in 0..3 {
                lab[atom][k] = cartesian[base + 3 * atom + k];
            }
        }
        let mut centre = [0.0; 3];
        let masses = [MASS_O, MASS_H, MASS_H];
        let mass = MASS_O + 2.0 * MASS_H;
        for atom in 0..3 {
            for k in 0..3 {
                centre[k] += masses[atom] * lab[atom][k];
            }
        }
        for value in centre.iter_mut() {
            *value /= mass;
        }
        let relative = lab.map(|point| {
            [
                point[0] - centre[0],
                point[1] - centre[1],
                point[2] - centre[2],
            ]
        });
        let rotation = kabsch(&body, &relative);
        let omega = rotation_vector(rotation);
        for k in 0..3 {
            state[3 * molecule + k] = centre[k];
            state[3 * n + 3 * molecule + k] = omega[k];
        }
    }
    state
}

/// Energy of Cartesian O, H, H water, with M placed on the bisector.
pub fn energy_ohh(cartesian: &[f64]) -> f64 {
    let placed = sites_from_ohh(cartesian);
    site_energy_forces(&placed).0
}

/// Interleaved GMIN angle-axis `(c, ω)` to block Rodrigues `(centres, -ω)`.
pub fn block_from_interleaved_gmin(interleaved: &[f64]) -> Vec<f64> {
    assert!(
        interleaved.len().is_multiple_of(6),
        "interleaved TIP4P state is 6 per molecule"
    );
    let n = interleaved.len() / 6;
    let mut block = vec![0.0; interleaved.len()];
    for molecule in 0..n {
        for k in 0..3 {
            block[3 * molecule + k] = interleaved[6 * molecule + k];
            block[3 * n + 3 * molecule + k] = -interleaved[6 * molecule + 3 + k];
        }
    }
    block
}

/// Block Rodrigues state to interleaved GMIN angle-axis.
pub fn interleaved_gmin_from_block(block: &[f64]) -> Vec<f64> {
    assert!(
        block.len().is_multiple_of(6),
        "block TIP4P state is 6 per molecule"
    );
    let n = block.len() / 6;
    let mut interleaved = vec![0.0; block.len()];
    for molecule in 0..n {
        for k in 0..3 {
            interleaved[6 * molecule + k] = block[3 * molecule + k];
            interleaved[6 * molecule + 3 + k] = -block[3 * n + 3 * molecule + k];
        }
    }
    interleaved
}

fn sites_from_ohh(cartesian: &[f64]) -> Vec<[[f64; 3]; 4]> {
    assert!(cartesian.len().is_multiple_of(9) && !cartesian.is_empty());
    let n = cartesian.len() / 9;
    let mut placed = Vec::with_capacity(n);
    for molecule in 0..n {
        let base = 9 * molecule;
        let mut atoms = [[0.0; 3]; 3];
        for atom in 0..3 {
            for k in 0..3 {
                atoms[atom][k] = cartesian[base + 3 * atom + k];
            }
        }
        let mut bond_1 = [0.0; 3];
        let mut bond_2 = [0.0; 3];
        for k in 0..3 {
            bond_1[k] = atoms[1][k] - atoms[0][k];
            bond_2[k] = atoms[2][k] - atoms[0][k];
        }
        let n1 = (bond_1[0] * bond_1[0] + bond_1[1] * bond_1[1] + bond_1[2] * bond_1[2]).sqrt();
        let n2 = (bond_2[0] * bond_2[0] + bond_2[1] * bond_2[1] + bond_2[2] * bond_2[2]).sqrt();
        for k in 0..3 {
            bond_1[k] /= n1;
            bond_2[k] /= n2;
        }
        let mut bisector = [
            bond_1[0] + bond_2[0],
            bond_1[1] + bond_2[1],
            bond_1[2] + bond_2[2],
        ];
        let nb =
            (bisector[0] * bisector[0] + bisector[1] * bisector[1] + bisector[2] * bisector[2])
                .sqrt();
        for value in bisector.iter_mut() {
            *value /= nb;
        }
        placed.push([
            atoms[0],
            atoms[1],
            atoms[2],
            [
                atoms[0][0] + R_OM * bisector[0],
                atoms[0][1] + R_OM * bisector[1],
                atoms[0][2] + R_OM * bisector[2],
            ],
        ]);
    }
    placed
}

fn site_energy_forces(placed: &[[[f64; 3]; 4]]) -> (f64, Vec<[[f64; 3]; 4]>) {
    let n = placed.len();
    let mut energy = 0.0;
    let mut force = vec![[[0.0; 3]; 4]; n];
    let charges = [0.0, Q_H, Q_H, Q_M];
    for left in 0..n {
        for right in (left + 1)..n {
            let delta = [
                placed[left][0][0] - placed[right][0][0],
                placed[left][0][1] - placed[right][0][1],
                placed[left][0][2] - placed[right][0][2],
            ];
            let distance2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            if distance2 > 0.0 {
                let inverse = 1.0 / distance2;
                let r6 = inverse * inverse * inverse;
                let r12 = r6 * r6;
                energy += C12 * r12 - C6 * r6;
                let slope = (-12.0 * C12 * r12 + 6.0 * C6 * r6) * inverse;
                for k in 0..3 {
                    force[left][0][k] += slope * delta[k];
                    force[right][0][k] -= slope * delta[k];
                }
            }
            for a in 1..4 {
                for b in 1..4 {
                    let delta = [
                        placed[left][a][0] - placed[right][b][0],
                        placed[left][a][1] - placed[right][b][1],
                        placed[left][a][2] - placed[right][b][2],
                    ];
                    let distance2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                    if distance2 <= 0.0 {
                        continue;
                    }
                    let distance = distance2.sqrt();
                    let pair = COULOMB * charges[a] * charges[b] / distance;
                    energy += pair;
                    let slope = -pair / distance2;
                    for k in 0..3 {
                        force[left][a][k] += slope * delta[k];
                        force[right][b][k] -= slope * delta[k];
                    }
                }
            }
        }
    }
    (energy, force)
}

fn kabsch(body: &[[f64; 3]; 3], lab: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut correlation = [[0.0; 3]; 3];
    for atom in 0..3 {
        for row in 0..3 {
            for col in 0..3 {
                correlation[row][col] += body[atom][row] * lab[atom][col];
            }
        }
    }
    // Right singular vectors of H. Three atoms are planar, so one singular
    // value is zero and the corresponding left vector has to be completed
    // by a cross product. Leaving that column zero makes a matrix the
    // exponential map cannot represent, and the sites come back reflected.
    let (evals, vectors) =
        sort_descending(jacobi_eigen(mat_mul(transpose(correlation), correlation)));
    let singular = evals.map(|value| value.max(0.0).sqrt());
    let mut left = [[0.0; 3]; 3];
    let mut defined = [false; 3];
    for index in 0..3 {
        if singular[index] <= 1e-8 {
            continue;
        }
        let axis = [vectors[0][index], vectors[1][index], vectors[2][index]];
        let mut column = mat_vec(correlation, axis);
        let norm = (column[0] * column[0] + column[1] * column[1] + column[2] * column[2]).sqrt();
        if norm <= 1e-12 {
            continue;
        }
        for value in column.iter_mut() {
            *value /= norm;
        }
        for row in 0..3 {
            left[row][index] = column[row];
        }
        defined[index] = true;
    }
    let known: Vec<usize> = (0..3).filter(|index| defined[*index]).collect();
    if known.len() >= 2 {
        let a = [left[0][known[0]], left[1][known[0]], left[2][known[0]]];
        let b = [left[0][known[1]], left[1][known[1]], left[2][known[1]]];
        let missing = (0..3).find(|index| !defined[*index]).unwrap_or(2);
        let column = cross(a, b);
        for row in 0..3 {
            left[row][missing] = column[row];
        }
    } else if known.len() == 1 {
        let a = [left[0][known[0]], left[1][known[0]], left[2][known[0]]];
        let helper = if a[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let b = cross(a, helper);
        let c = cross(a, b);
        let mut slot = 0usize;
        for index in 0..3 {
            if defined[index] {
                continue;
            }
            let column = if slot == 0 { b } else { c };
            for row in 0..3 {
                left[row][index] = column[row];
            }
            slot += 1;
        }
    } else {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    if determinant(mat_mul(vectors, transpose(left))) < 0.0 {
        for row in 0..3 {
            left[row][2] = -left[row][2];
        }
    }
    mat_mul(vectors, transpose(left))
}

fn jacobi_eigen(mut matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut best = matrix[0][1].abs();
        for (row, col, magnitude) in [(0, 2, matrix[0][2].abs()), (1, 2, matrix[1][2].abs())] {
            if magnitude > best {
                best = magnitude;
                p = row;
                q = col;
            }
        }
        if best < 1e-14 {
            break;
        }
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let mut next = matrix;
        next[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        next[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        next[p][q] = 0.0;
        next[q][p] = 0.0;
        for row in 0..3 {
            if row == p || row == q {
                continue;
            }
            let arp = matrix[row][p];
            let arq = matrix[row][q];
            next[row][p] = cosine * arp - sine * arq;
            next[p][row] = next[row][p];
            next[row][q] = sine * arp + cosine * arq;
            next[q][row] = next[row][q];
        }
        matrix = next;
        for row in 0..3 {
            let vip = vectors[row][p];
            let viq = vectors[row][q];
            vectors[row][p] = cosine * vip - sine * viq;
            vectors[row][q] = sine * vip + cosine * viq;
        }
    }
    ([matrix[0][0], matrix[1][1], matrix[2][2]], vectors)
}

fn sort_descending(eigen: ([f64; 3], [[f64; 3]; 3])) -> ([f64; 3], [[f64; 3]; 3]) {
    let (values, vectors) = eigen;
    let mut order = [0usize, 1, 2];
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    let sorted = [values[order[0]], values[order[1]], values[order[2]]];
    let mut arranged = [[0.0; 3]; 3];
    for col in 0..3 {
        for row in 0..3 {
            arranged[row][col] = vectors[row][order[col]];
        }
    }
    (sorted, arranged)
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mat_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = left[row][0] * right[0][col]
                + left[row][1] * right[1][col]
                + left[row][2] * right[2][col];
        }
    }
    out
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::warm_lbfgs::WarmLbfgs;

    fn load_xyz(text: &str) -> (f64, Vec<f64>) {
        let mut lines = text.lines();
        let _count = lines.next();
        let header = lines.next().unwrap_or("");
        let energy = header
            .split('=')
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|token| token.parse::<f64>().ok())
            .expect("header carries the reference energy");
        let mut coords = Vec::new();
        for line in lines {
            let mut parts = line.split_whitespace();
            let _symbol = parts.next();
            for _ in 0..3 {
                coords.push(parts.next().unwrap().parse::<f64>().unwrap());
            }
        }
        (energy, coords)
    }

    #[test]
    fn cartesian_minima_match_the_reference_energies() {
        let cases = [
            (
                include_str!("../../tests/fixtures/tip4p-10.xyz"),
                -391.0226842125,
            ),
            (
                include_str!("../../tests/fixtures/tip4p-15.xyz"),
                -628.3722525077,
            ),
            (
                include_str!("../../tests/fixtures/tip4p-20.xyz"),
                -872.9887544952,
            ),
        ];
        for (text, published) in cases {
            let (header, coords) = load_xyz(text);
            assert!((header - published).abs() < 1e-9);
            let energy = energy_ohh(&coords);
            let rigid = rigid_from_ohh(&coords);
            let n = coords.len() / 9;
            let potential = GminTip4p::new(n);
            let (rigid_energy, gradient) = potential.value_and_gradient(rigid.view());
            let gmax = gradient
                .iter()
                .fold(0.0_f64, |best, value| best.max(value.abs()));
            // The xyz files store coordinates to 7 decimals. The header is
            // the published minimum of a finer geometry, so the rigid
            // energy of this file sits a few thousandths below it. A wrong
            // charge or M site misses by a kilojoule, not by that margin.
            assert!(
                (rigid_energy - published).abs() < 5.0e-3,
                "rigid energy {rigid_energy} against {published}"
            );
            assert!(
                rigid_energy <= published + 1.0e-3,
                "rigid energy {rigid_energy} is above the success threshold"
            );
            assert!(
                (energy - rigid_energy).abs() < 1.0e-3,
                "site energy {energy} against rigid {rigid_energy}"
            );
            assert!(
                gmax < 1.0e-3,
                "reconstructed minimum gradient {gmax} is above the record tolerance"
            );
        }
    }

    #[test]
    fn the_rigid_gradient_matches_a_finite_difference() {
        let potential = GminTip4p::new(2);
        let mut state = Array1::zeros(12);
        state[0] = 0.1;
        state[3] = 3.0;
        state[4] = 0.2;
        state[6] = 0.4;
        state[7] = -0.2;
        state[8] = 0.1;
        state[9] = 0.3;
        state[11] = -0.5;
        let (_, gradient) = potential.value_and_gradient(state.view());
        let step = 1e-6;
        for k in 0..state.len() {
            let mut plus = state.clone();
            let mut minus = state.clone();
            plus[k] += step;
            minus[k] -= step;
            let numeric =
                (potential.eval(plus.view()) - potential.eval(minus.view())) / (2.0 * step);
            let scale = 1.0 + numeric.abs();
            assert!(
                (gradient[k] - numeric).abs() / scale < 1e-5,
                "coordinate {k}: analytic {} against numeric {numeric}",
                gradient[k]
            );
        }
    }

    #[test]
    fn a_shifted_ten_water_minimum_quenches_back() {
        let (published, coords) = load_xyz(include_str!("../../tests/fixtures/tip4p-10.xyz"));
        let start = rigid_from_ohh(&coords);
        let potential = GminTip4p::new(10);
        let unperturbed = potential.eval(start.view());
        let mut shifted = start.clone();
        shifted[0] += 0.15;
        shifted[3 * 10] += 0.2;
        let mut opt = WarmLbfgs::default();
        opt.gtol = 1e-3;
        let (energy, _, evals) = opt.minimize(shifted.view(), 80, |point| {
            Some(potential.value_and_gradient(point))
        });
        assert!(
            energy <= published + 1.0e-3,
            "quench energy {energy} after {evals} evaluations, reference {published}"
        );
        assert!(
            (energy - unperturbed).abs() < 1.0e-3,
            "quench energy {energy} left the basin of {unperturbed}"
        );
    }

    #[test]
    fn interleaved_angle_axis_round_trips() {
        let block = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.2, -0.3, 0.4, 0.5, 0.1, -0.2];
        let interleaved = interleaved_gmin_from_block(&block);
        let back = block_from_interleaved_gmin(&interleaved);
        for (left, right) in block.iter().zip(back.iter()) {
            assert!((left - right).abs() < 1e-15);
        }
        assert!((interleaved[3] + block[6]).abs() < 1e-15);
    }
}
