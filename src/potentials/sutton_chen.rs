//! Sutton–Chen metal potential in the reduced units of the Cambridge tables.
//!
//! Doye and Wales, *New J. Chem.* **22**, 733 (1998), set `ε = 1` and `a = 1`
//! and use Sutton and Chen's dimensionless `c`. The 12–6 row is silver and
//! rhodium (`c = 144.41`); the 9–6 row is nickel and copper (`c = 39.432`).
//!
//! ```text
//! E = ½ Σ_i Σ_{j≠i} (a/r_ij)^n − c Σ_i √ρ_i
//! ρ_i = Σ_{j≠i} (a/r_ij)^m
//! ```

use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

/// Which Sutton–Chen parameter row a cluster is bound by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuttonChenFamily {
    /// `n = 12`, `m = 6`, `c = 144.41`. Silver and rhodium.
    Sc12_6,
    /// `n = 9`, `m = 6`, `c = 39.432`. Nickel and copper.
    Sc9_6,
}

impl SuttonChenFamily {
    /// Parse `12-6` or `9-6`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "12-6" | "12_6" | "sc12-6" => Some(Self::Sc12_6),
            "9-6" | "9_6" | "sc9-6" => Some(Self::Sc9_6),
            _ => None,
        }
    }

    /// Repulsive exponent `n`.
    pub fn n_exp(self) -> i32 {
        match self {
            Self::Sc12_6 => 12,
            Self::Sc9_6 => 9,
        }
    }

    /// Density exponent `m`.
    pub fn m_exp(self) -> i32 {
        6
    }

    /// Sutton–Chen `c`, exactly the value printed on the Cambridge point files.
    pub fn c(self) -> f64 {
        match self {
            Self::Sc12_6 => 144.41,
            Self::Sc9_6 => 39.432,
        }
    }

    /// Reduced temperature `kT/ε` used by Doye and Wales for basin-hopping.
    pub fn basin_hopping_temperature(self) -> f64 {
        match self {
            Self::Sc12_6 => 30.0,
            Self::Sc9_6 => 10.0,
        }
    }

    /// Name used in logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::Sc12_6 => "12-6",
            Self::Sc9_6 => "9-6",
        }
    }
}

/// Sutton–Chen cluster over `n` free points, flat point-major coordinates.
#[derive(Debug, Clone)]
pub struct SuttonChen {
    /// Points in a configuration.
    pub n_points: usize,
    family: SuttonChenFamily,
    bounds: Bounds<f64>,
}

impl SuttonChen {
    /// Potential over `n_points` in a box wide enough for a compact cluster.
    pub fn new(n_points: usize, family: SuttonChenFamily) -> Self {
        assert!(n_points >= 2, "a Sutton-Chen cluster needs at least two points");
        let dim = 3 * n_points;
        // Bulk nearest-neighbour distance at lattice constant a = 1 is 1/√2.
        let nn = std::f64::consts::FRAC_1_SQRT_2;
        let extent = 4.0 * nn * (n_points as f64).cbrt();
        Self {
            n_points,
            family,
            bounds: Bounds::new(
                Array1::from_elem(dim, -extent),
                Array1::from_elem(dim, extent),
                0.0,
            ),
        }
    }

    /// The parameter row.
    pub fn family(&self) -> SuttonChenFamily {
        self.family
    }

    /// Energy and Cartesian gradient.
    ///
    /// The gradient matches the analytic derivative in GMIN's `sc.f90`: the
    /// pair term contributes `−n V_ij (x_i − x_j) / r^2` and the embedding
    /// term `m c φ_ij (ρ_i^{−1/2} + ρ_j^{−1/2}) / 2 · (x_i − x_j) / r^2`.
    pub fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_points;
        let n_exp = self.family.n_exp();
        let m_exp = self.family.m_exp();
        let c = self.family.c();
        let mut rho = vec![0.0; n];
        let mut pair = 0.0;
        // Store (j, i, φ, V, dx, dy, dz, inv_r2) only while differentiating.
        // One pass builds ρ and the pair energy; a second applies forces.
        let mut bonds: Vec<Bond> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[3 * i] - x[3 * j];
                let dy = x[3 * i + 1] - x[3 * j + 1];
                let dz = x[3 * i + 2] - x[3 * j + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 <= 1e-24 {
                    continue;
                }
                let (v, phi) = reduced_powers(r2, n_exp, m_exp);
                pair += v;
                rho[i] += phi;
                rho[j] += phi;
                bonds.push(Bond {
                    i,
                    j,
                    phi,
                    v,
                    dx,
                    dy,
                    dz,
                    inv_r2: 1.0 / r2,
                });
            }
        }
        let mut sqrt_rho = vec![0.0; n];
        let mut inv_sqrt = vec![0.0; n];
        let mut emb = 0.0;
        for i in 0..n {
            if rho[i] > 0.0 {
                let s = rho[i].sqrt();
                sqrt_rho[i] = s;
                inv_sqrt[i] = 1.0 / s;
                emb += s;
            }
        }
        let energy = pair - c * emb;
        let mut g = Array1::zeros(x.len());
        let n_exp_f = n_exp as f64;
        let m_exp_f = m_exp as f64;
        for bond in &bonds {
            let embed = m_exp_f * c * bond.phi * (inv_sqrt[bond.i] + inv_sqrt[bond.j]) * 0.5;
            let coef = (embed - n_exp_f * bond.v) * bond.inv_r2;
            let d = [bond.dx, bond.dy, bond.dz];
            for k in 0..3 {
                let f = coef * d[k];
                g[3 * bond.i + k] += f;
                g[3 * bond.j + k] -= f;
            }
        }
        let _ = sqrt_rho;
        (energy, g)
    }
}

struct Bond {
    i: usize,
    j: usize,
    phi: f64,
    v: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    inv_r2: f64,
}

/// `(r^{−n}, r^{−m})` at squared separation `r2`, with `a = 1`.
fn reduced_powers(r2: f64, n_exp: i32, m_exp: i32) -> (f64, f64) {
    let inv2 = 1.0 / r2;
    let phi = if m_exp == 6 {
        let inv3 = inv2 * inv2 * inv2;
        inv3
    } else {
        r2.sqrt().powi(-m_exp)
    };
    let v = match n_exp {
        12 => phi * phi,
        9 => phi * inv2 / r2.sqrt(),
        _ => r2.sqrt().powi(-n_exp),
    };
    (v, phi)
}

impl Objective<f64> for SuttonChen {
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

impl Gradient<f64> for SuttonChen {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        3 * self.n_points
    }
}

impl DifferentiableObjective<f64> for SuttonChen {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        SuttonChen::value_and_gradient(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two atoms: `E = r^{−n} − 2 c r^{−m/2}`.
    #[test]
    fn a_dimer_matches_the_closed_form() {
        let r = 0.7;
        let p = SuttonChen::new(2, SuttonChenFamily::Sc12_6);
        let x = Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]);
        let (e, _) = p.value_and_gradient(x.view());
        let c = SuttonChenFamily::Sc12_6.c();
        let closed = r.powi(-12) - 2.0 * c * r.powi(-3);
        assert!((e - closed).abs() < 1e-8, "{e} vs {closed}");
    }

    #[test]
    fn the_gradient_matches_a_central_difference() {
        let p = SuttonChen::new(4, SuttonChenFamily::Sc9_6);
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.2, 0.75, 0.15, -0.3, 0.4, 0.7,
        ]);
        let (e0, g) = p.value_and_gradient(x.view());
        let eps = 1e-6;
        for k in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += eps;
            xm[k] -= eps;
            let num = (p.eval(xp.view()) - p.eval(xm.view())) / (2.0 * eps);
            assert!(
                (num - g[k]).abs() < 1e-4,
                "component {k}: numeric {num} analytic {} energy {e0}",
                g[k]
            );
        }
    }
}
