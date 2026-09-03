//! Rigid Jorgensen TIP4P water as a cluster objective.
//!
//! Each molecule is a rigid body. The state is `6 n` coordinates, molecule
//! major: the `n` centres of mass, then an exponential-map rotation vector
//! `ω` for each molecule. `R(ω) = exp([ω]×)` via Rodrigues' formula. That
//! chart is a local diffeomorphism for `|ω| < 2π`, so the energy has a
//! gradient with respect to the rotation coordinates. A unit quaternion
//! would need a projected gradient on `S³`; the exponential map keeps the
//! dimension at six per molecule and matches the hopping chain's
//! three-coordinates-per-point layout on the translational half of the
//! state.
//!
//! Site geometry and parameters follow Jorgensen, Chandrasekhar, Madura,
//! Impey and Klein, *J. Chem. Phys.* **79**, 926 (1983). There is no cutoff:
//! every intermolecular pair contributes. Energy is in kJ/mol. The analytic
//! gradient is assembled from site forces and torques: the centre-of-mass
//! block is the total force, and `∂E/∂ω = J_r(ω)ᵀ Σ (R s) × (∂E/∂r)` with
//! the right Jacobian of the exponential map.

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

/// O–H bond length, angstrom.
pub const R_OH: f64 = 0.9572;
/// H–O–H angle, degrees.
pub const HOH_DEG: f64 = 104.52;
/// Dummy-site distance from oxygen along the HOH bisector, angstrom.
pub const R_OM: f64 = 0.15;
/// Hydrogen charge, elementary charges.
pub const Q_H: f64 = 0.52;
/// M-site charge, elementary charges.
pub const Q_M: f64 = -1.04;
/// Oxygen Lennard-Jones well depth, kJ/mol.
pub const EPSILON: f64 = 0.6485;
/// Oxygen Lennard-Jones diameter, angstrom.
pub const SIGMA: f64 = 3.15365;
/// Coulomb constant, kJ/mol Å e⁻².
pub const COULOMB: f64 = 1389.35458;
/// Oxygen mass used to place the centre of mass, u.
const MASS_O: f64 = 15.9994;
/// Hydrogen mass used to place the centre of mass, u.
const MASS_H: f64 = 1.008;

/// Wales and Hodges, *Chem. Phys. Lett.* **286**, 65 (1998), putative
/// TIP4P global minima in kJ/mol.
pub fn wales_hodges_minimum(n_molecules: usize) -> Option<f64> {
    Some(match n_molecules {
        2 => -26.08757,
        6 => -197.78,
        8 => -305.52,
        10 => -391.02,
        12 => -492.91,
        20 => -872.99,
        21 => -916.71,
        _ => return None,
    })
}

/// Rigid TIP4P cluster of `n_molecules` waters.
#[derive(Debug, Clone)]
pub struct Tip4pCluster {
    /// Waters in the cluster.
    pub n_molecules: usize,
    bounds: Bounds<f64>,
    /// Body-frame site positions relative to the centre of mass: O, H, H, M.
    body: [[f64; 3]; 4],
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl Tip4pCluster {
    /// A cluster of `n_molecules` rigid TIP4P waters.
    pub fn new(n_molecules: usize) -> Self {
        assert!(
            n_molecules >= 2,
            "a TIP4P cluster needs at least two molecules"
        );
        let dim = 6 * n_molecules;
        let extent = 4.0 * SIGMA * (n_molecules as f64).cbrt();
        Self {
            n_molecules,
            bounds: Bounds::new(
                Array1::from_elem(dim, -extent),
                Array1::from_elem(dim, extent),
                0.0,
            ),
            body: body_sites(),
        }
    }

    /// Coordinate dimension: three centre-of-mass and three rotation numbers
    /// per molecule.
    pub fn dim_coords(&self) -> usize {
        6 * self.n_molecules
    }

    /// Fold every rotation vector into the ball of radius `π`.
    ///
    /// `R(ω)` is `2π`-periodic along a ray. Keeping the chart in the
    /// principal ball keeps the right Jacobian away from the singularities
    /// at `|ω| = 2π k`.
    pub fn fold_rotations(&self, x: &mut [f64]) {
        fold_rotation_vectors(self.n_molecules, x);
    }

    /// Lab-frame sites for molecule `i`: O, H, H, M.
    pub fn molecule_sites(&self, x: ArrayView1<f64>, i: usize) -> [[f64; 3]; 4] {
        let n = self.n_molecules;
        let com = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
        let w = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
        let rot = rotation_matrix(w);
        let mut sites = [[0.0; 3]; 4];
        for (s, body) in sites.iter_mut().zip(self.body.iter()) {
            let p = mat_vec(rot, *body);
            *s = [com[0] + p[0], com[1] + p[1], com[2] + p[2]];
        }
        sites
    }

    /// Energy and rigid-body gradient in one pass.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_molecules;
        assert_eq!(x.len(), 6 * n, "TIP4P state is 6 coordinates per molecule");
        let mut sites = vec![[[0.0; 3]; 4]; n];
        let mut rots = vec![IDENTITY; n];
        for i in 0..n {
            let com = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
            let w = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
            rots[i] = rotation_matrix(w);
            for (s, body) in sites[i].iter_mut().zip(self.body.iter()) {
                let p = mat_vec(rots[i], *body);
                *s = [com[0] + p[0], com[1] + p[1], com[2] + p[2]];
            }
        }

        let mut energy = 0.0;
        let mut force = vec![[[0.0; 3]; 4]; n];
        let charges = [0.0, Q_H, Q_H, Q_M];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    sites[i][0][0] - sites[j][0][0],
                    sites[i][0][1] - sites[j][0][1],
                    sites[i][0][2] - sites[j][0][2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 > 0.0 {
                    let inv2 = SIGMA * SIGMA / r2;
                    let inv6 = inv2 * inv2 * inv2;
                    let inv12 = inv6 * inv6;
                    energy += 4.0 * EPSILON * (inv12 - inv6);
                    let coef = -24.0 * EPSILON / r2 * (2.0 * inv12 - inv6);
                    for k in 0..3 {
                        force[i][0][k] += coef * d[k];
                        force[j][0][k] -= coef * d[k];
                    }
                }
                for a in 1..4 {
                    for b in 1..4 {
                        let d = [
                            sites[i][a][0] - sites[j][b][0],
                            sites[i][a][1] - sites[j][b][1],
                            sites[i][a][2] - sites[j][b][2],
                        ];
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 <= 0.0 {
                            continue;
                        }
                        let r = r2.sqrt();
                        let qq = COULOMB * charges[a] * charges[b];
                        energy += qq / r;
                        let coef = -qq / (r2 * r);
                        for k in 0..3 {
                            force[i][a][k] += coef * d[k];
                            force[j][b][k] -= coef * d[k];
                        }
                    }
                }
            }
        }

        let mut g = Array1::zeros(6 * n);
        for i in 0..n {
            let mut torque = [0.0; 3];
            for a in 0..4 {
                for k in 0..3 {
                    g[3 * i + k] += force[i][a][k];
                }
                let rel = mat_vec(rots[i], self.body[a]);
                let t = cross(rel, force[i][a]);
                for k in 0..3 {
                    torque[k] += t[k];
                }
            }
            let w = [x[3 * n + 3 * i], x[3 * n + 3 * i + 1], x[3 * n + 3 * i + 2]];
            let jr_t = right_jacobian_t(w);
            let gw = mat_vec(jr_t, torque);
            for k in 0..3 {
                g[3 * n + 3 * i + k] = gw[k];
            }
        }
        (energy, g)
    }
}

impl Objective<f64> for Tip4pCluster {
    fn dim(&self) -> usize {
        self.dim_coords()
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Tip4pCluster {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        self.dim_coords()
    }
}

impl DifferentiableObjective<f64> for Tip4pCluster {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        Tip4pCluster::value_and_gradient(self, x)
    }
}

impl crate::pes_exploration::PesSurface for Tip4pCluster {
    type Error = std::convert::Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        Ok(self.value_and_gradient(coordinates))
    }
}

/// Random compact cluster: centres of mass at liquid-like density, random
/// orientations in the principal rotation ball.
pub fn random_cluster<R: rand::Rng + ?Sized>(n_molecules: usize, rng: &mut R) -> Array1<f64> {
    let radius = 1.15 * SIGMA * (n_molecules as f64).cbrt();
    let min_sep = 2.35;
    let mut coms: Vec<[f64; 3]> = Vec::with_capacity(n_molecules);
    let mut tries = 0;
    while coms.len() < n_molecules && tries < 40_000 {
        tries += 1;
        let mut v = [0.0; 3];
        let mut norm = 0.0;
        for k in 0..3 {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            v[k] = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            norm += v[k] * v[k];
        }
        let norm = norm.sqrt().max(1e-12);
        let r = radius * rng.random::<f64>().cbrt();
        let p = [v[0] / norm * r, v[1] / norm * r, v[2] / norm * r];
        if coms.iter().all(|q| {
            let d0 = p[0] - q[0];
            let d1 = p[1] - q[1];
            let d2 = p[2] - q[2];
            (d0 * d0 + d1 * d1 + d2 * d2).sqrt() >= min_sep
        }) {
            coms.push(p);
        }
    }
    assert_eq!(
        coms.len(),
        n_molecules,
        "could not place {n_molecules} waters without overlap"
    );
    let mut x = Array1::zeros(6 * n_molecules);
    for (i, p) in coms.iter().enumerate() {
        for k in 0..3 {
            x[3 * i + k] = p[k];
        }
        for k in 0..3 {
            x[3 * n_molecules + 3 * i + k] =
                rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        }
    }
    fold_rotation_vectors(n_molecules, x.as_slice_mut().expect("contiguous state"));
    x
}

/// Fold each exponential-map rotation vector into the ball of radius `π`.
pub fn fold_rotation_vectors(n_molecules: usize, x: &mut [f64]) {
    fold_rotations(n_molecules, x);
}

/// Jorgensen linear dimer: donor O at the origin, acceptor O on `+z` at
/// 2.75 Å, acceptor bisector tilted 46° from the O–O axis.
pub fn jorgensen_dimer() -> Array1<f64> {
    let body = body_sites();
    let oh = [
        body[1][0] - body[0][0],
        body[1][1] - body[0][1],
        body[1][2] - body[0][2],
    ];
    let donor_r = rotation_taking(oh, [0.0, 0.0, 1.0]);
    let donor_w = rotation_vector(donor_r);
    let donor_o = mat_vec(donor_r, body[0]);
    let donor_com = [-donor_o[0], -donor_o[1], -donor_o[2]];

    let tilt = 46.0_f64.to_radians();
    let acc_r = rotation_matrix([0.0, tilt, 0.0]);
    let acc_o = mat_vec(acc_r, body[0]);
    let acc_com = [-acc_o[0], -acc_o[1], 2.75 - acc_o[2]];
    let acc_w = rotation_vector(acc_r);

    let mut x = Array1::zeros(12);
    for k in 0..3 {
        x[k] = donor_com[k];
        x[3 + k] = acc_com[k];
        x[6 + k] = donor_w[k];
        x[9 + k] = acc_w[k];
    }
    x
}

fn body_sites() -> [[f64; 3]; 4] {
    let half = (HOH_DEG * 0.5).to_radians();
    let (s, c) = half.sin_cos();
    let h1 = [R_OH * s, 0.0, R_OH * c];
    let h2 = [-R_OH * s, 0.0, R_OH * c];
    let m = [0.0, 0.0, R_OM];
    let o = [0.0, 0.0, 0.0];
    let mass = MASS_O + 2.0 * MASS_H;
    let com_z = 2.0 * MASS_H * h1[2] / mass;
    [
        [o[0], o[1], o[2] - com_z],
        [h1[0], h1[1], h1[2] - com_z],
        [h2[0], h2[1], h2[2] - com_z],
        [m[0], m[1], m[2] - com_z],
    ]
}

fn fold_rotations(n: usize, x: &mut [f64]) {
    for i in 0..n {
        let b = 3 * n + 3 * i;
        let mut w = [x[b], x[b + 1], x[b + 2]];
        loop {
            let theta = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            if theta <= std::f64::consts::PI || theta == 0.0 {
                break;
            }
            let scale = 1.0 - 2.0 * std::f64::consts::PI / theta;
            w[0] *= scale;
            w[1] *= scale;
            w[2] *= scale;
        }
        x[b] = w[0];
        x[b + 1] = w[1];
        x[b + 2] = w[2];
    }
}

fn hat(w: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mat_add(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

fn mat_scale(s: f64, a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = s * a[i][j];
        }
    }
    c
}

fn mat_vec(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn mat_transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn rotation_matrix(w: [f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if theta2 < 1e-16 {
        return IDENTITY;
    }
    let theta = theta2.sqrt();
    let k = hat(w);
    let k2 = mat_mul(k, k);
    let (sin, cos) = theta.sin_cos();
    let a = if theta < 1e-4 {
        1.0 - theta2 / 6.0
    } else {
        sin / theta
    };
    let b = if theta < 1e-4 {
        0.5 - theta2 / 24.0
    } else {
        (1.0 - cos) / theta2
    };
    mat_add(IDENTITY, mat_add(mat_scale(a, k), mat_scale(b, k2)))
}

/// Right Jacobian transpose of the exponential map, so `∂E/∂ω = J_rᵀ τ`
/// when `τ` is the body-to-space torque assembled from site forces.
fn right_jacobian_t(w: [f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if theta2 < 1e-16 {
        return IDENTITY;
    }
    let theta = theta2.sqrt();
    let k = hat(w);
    let k2 = mat_mul(k, k);
    let (sin, cos) = theta.sin_cos();
    let alpha = if theta < 1e-4 {
        0.5 - theta2 / 24.0
    } else {
        (1.0 - cos) / theta2
    };
    let beta = if theta < 1e-4 {
        1.0 / 6.0 - theta2 / 120.0
    } else {
        (theta - sin) / (theta2 * theta)
    };
    // J_r = I − α [ω]× + β [ω]×²
    let jr = mat_add(IDENTITY, mat_add(mat_scale(-alpha, k), mat_scale(beta, k2)));
    mat_transpose(jr)
}

fn rotation_taking(from: [f64; 3], to: [f64; 3]) -> [[f64; 3]; 3] {
    let nf = (from[0] * from[0] + from[1] * from[1] + from[2] * from[2])
        .sqrt()
        .max(1e-15);
    let nt = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2])
        .sqrt()
        .max(1e-15);
    let a = [from[0] / nf, from[1] / nf, from[2] / nf];
    let b = [to[0] / nt, to[1] / nt, to[2] / nt];
    let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if c > 1.0 - 1e-14 {
        return IDENTITY;
    }
    if c < -1.0 + 1e-14 {
        let mut axis = cross(a, [1.0, 0.0, 0.0]);
        let n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        if n2 < 1e-12 {
            axis = cross(a, [0.0, 1.0, 0.0]);
        }
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        return rotation_matrix([
            axis[0] / n * std::f64::consts::PI,
            axis[1] / n * std::f64::consts::PI,
            axis[2] / n * std::f64::consts::PI,
        ]);
    }
    let v = cross(a, b);
    let k = hat(v);
    let k2 = mat_mul(k, k);
    mat_add(IDENTITY, mat_add(k, mat_scale(1.0 / (1.0 + c), k2)))
}

fn rotation_vector(r: [[f64; 3]; 3]) -> [f64; 3] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos.acos();
    if theta < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    let s = 2.0 * theta.sin();
    if s.abs() < 1e-12 {
        return [theta, 0.0, 0.0];
    }
    [
        (r[2][1] - r[1][2]) / s * theta,
        (r[0][2] - r[2][0]) / s * theta,
        (r[1][0] - r[0][1]) / s * theta,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::BasinBias;
    use crate::methods::cluster_hopping::{ClusterFingerprint, Config, Ledger, run_with_bias};
    use crate::methods::warm_lbfgs::WarmLbfgs;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn the_jorgensen_dimer_sits_near_minus_twenty_six() {
        let pot = Tip4pCluster::new(2);
        let x = jorgensen_dimer();
        let e = pot.eval(x.view());
        assert!(
            (e + 26.0).abs() < 1.5,
            "TIP4P dimer at the linear Jorgensen geometry is {e} kJ/mol"
        );
    }

    #[test]
    fn the_rigid_body_gradient_matches_a_finite_difference() {
        let pot = Tip4pCluster::new(3);
        let mut x = Array1::<f64>::zeros(18);
        let places = [[0.0, 0.0, 0.0], [2.85, 0.15, 0.05], [1.35, 2.55, -0.20]];
        for i in 0..3 {
            for k in 0..3 {
                x[3 * i + k] = places[i][k];
            }
            x[9 + 3 * i] = 0.31 * (i as f64 + 1.0);
            x[9 + 3 * i + 1] = -0.17 * (i as f64 + 0.4);
            x[9 + 3 * i + 2] = 0.22 * (i as f64 - 0.6);
        }
        let (_, g) = pot.value_and_gradient(x.view());
        let h = 1e-6;
        for k in 0..18 {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[k] += h;
            minus[k] -= h;
            let num = (pot.eval(plus.view()) - pot.eval(minus.view())) / (2.0 * h);
            let denom = 1.0 + num.abs();
            let rel = (g[k] - num).abs() / denom;
            assert!(
                rel < 1e-6,
                "coordinate {k}: analytic {} against numeric {num}, relative {rel}",
                g[k]
            );
        }
    }

    #[test]
    fn the_trait_methods_agree_with_the_fused_one() {
        let pot = Tip4pCluster::new(2);
        let x = jorgensen_dimer();
        let (e, g) = pot.value_and_gradient(x.view());
        assert_eq!(e, pot.eval(x.view()));
        assert_eq!(g, pot.grad(x.view()));
        assert_eq!(Objective::dim(&pot), 12);
        assert_eq!(Gradient::dim(&pot), 12);
    }

    #[test]
    fn the_hexamer_search_reaches_the_wales_hodges_minimum() {
        let pot = Tip4pCluster::new(6);
        let cfg = Config::for_tip4p(6);
        let target = wales_hodges_minimum(6).expect("hexamer table");
        let mut hits = 0usize;
        for seed in 0u64..4 {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15) + 3);
            let start = random_cluster(6, &mut rng);
            let mut ledger = Ledger::new(20_000);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, mut xr, _) = opt.minimize(x, iters, |v| {
                    if !led.charge() {
                        return None;
                    }
                    Some(pot.value_and_gradient(v))
                });
                pot.fold_rotations(xr.as_slice_mut().expect("contiguous"));
                (f, xr)
            };
            let mut bias = BasinBias::new(
                ClusterFingerprint::of_config(&cfg, &start),
                cfg.merge_radius,
                cfg.bias_height,
                cfg.bias_gamma,
            );
            let out = run_with_bias(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                None,
                &mut bias,
                &mut rng,
            );
            if out.best < target + 0.01 {
                hits += 1;
            }
        }
        assert!(
            hits >= 3,
            "hexamer search reached {target} on {hits}/4 seeds"
        );
    }
}
