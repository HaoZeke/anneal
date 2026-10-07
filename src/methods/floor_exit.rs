//! Recommended search that pulls the outer radius in and the inertia toward a sphere.
//!
//! The radial term is a smooth maximum of the distance from the centre of
//! mass. Its width is the entering cluster's nearest-neighbour distance
//! divided by the square root of the number of atoms. The angular term is
//! the squared spread of the three principal moments. On the entering
//! cluster each term equals the cohesive energy times the number of atoms.
//! The second step minimises the plain energy. No pair distance is
//! penalised, and no target energy is read.

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use crate::methods::warm_lbfgs::WarmLbfgs;

/// Recommended hops from `origin`, relaxed toward a spherical inertia tensor.
///
/// Returns the lowest plain energy seen.
pub fn search<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    _hops: usize,
    seed: u64,
    mut evaluate: E,
    quench: Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64> + Send,
{
    let _ = (contact, quench);
    let (start_energy, _) = evaluate(origin);
    let n = origin.len() / 3;
    if n < 2 || !start_energy.is_finite() {
        return start_energy;
    }
    let target = start_energy.abs() * n as f64;
    let width = radial_width(origin);
    let aniso0 = shape_parts(origin).0;
    let radius0 = smooth_radius(origin, width).0;
    let nu = if aniso0 < 1.0e-8 {
        0.0
    } else {
        target / aniso0
    };
    let mu = if radius0 < 1.0e-8 {
        0.0
    } else {
        target / radius0
    };
    let cfg = Config::recommended(n);
    let budget = n.saturating_mul(4_000);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut hop_index = 0usize;
    let mut best = start_energy;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        hop_index = hop_index.saturating_add(1);
        let hop = hop_index;
        opt.forget();
        let (_, bent, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            let (energy, gradient) = evaluate(v);
            let (extra_a, grad_a) = anisotropy_penalty(v, nu);
            let (extra_r, grad_r) = smooth_radius_penalty(v, width, mu);
            Some((energy + extra_a + extra_r, gradient + grad_a + grad_r))
        });
        opt.forget();
        let (mut energy, mut quenched, _) = opt.minimize(bent.view(), iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(evaluate(v))
        });
        if energy < start_energy - 1.0e-2 {
            opt.forget();
            let (polished, coords, _) =
                opt.minimize(quenched.view(), iters.saturating_mul(4), |v| {
                    if !led.charge() {
                        return None;
                    }
                    Some(evaluate(v))
                });
            energy = polished;
            quenched = coords;
        }
        if energy.is_finite() {
            println!(
                "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if energy < best {
                best = energy;
            }
            if energy < start_energy - 1.0e-2 {
                let _ = led.charge_many(led.remaining());
            }
        }
        (energy, quenched)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let out = run_with_gradient(&cfg, origin, &mut ledger, &mut relax, None, &mut rng);
    if out.best < best {
        best = out.best;
    }
    best
}

fn radial_width(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let shortest = shortest_pair(x);
    if n < 2 || !(shortest.is_finite() && shortest > 0.0) {
        return 1.0;
    }
    shortest / (n as f64).sqrt()
}

fn shortest_pair(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut best = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            if r2 < best {
                best = r2;
            }
        }
    }
    best.sqrt()
}

/// Smooth maximum of the distance from the centre of mass.
fn smooth_radius(x: ArrayView1<f64>, width: f64) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut gradient = Array1::zeros(x.len());
    if n == 0 || !(width.is_finite() && width > 0.0) {
        return (0.0, gradient);
    }
    let mut com = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            com[k] += x[3 * i + k];
        }
    }
    let scale = n as f64;
    for value in &mut com {
        *value /= scale;
    }
    let mut radii = vec![0.0; n];
    let mut zmax = f64::NEG_INFINITY;
    for i in 0..n {
        let mut r2 = 0.0;
        for k in 0..3 {
            let d = x[3 * i + k] - com[k];
            r2 += d * d;
        }
        radii[i] = r2.sqrt();
        zmax = zmax.max(radii[i] / width);
    }
    if !zmax.is_finite() {
        return (0.0, gradient);
    }
    let mut sum = 0.0;
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let weight = (radii[i] / width - zmax).exp();
        weights[i] = weight;
        sum += weight;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return (0.0, gradient);
    }
    let value = width * (zmax + sum.ln());
    let mut mean = [0.0; 3];
    for i in 0..n {
        let share = weights[i] / sum;
        let radius = radii[i].max(1.0e-12);
        for k in 0..3 {
            let unit = (x[3 * i + k] - com[k]) / radius;
            let piece = share * unit;
            gradient[3 * i + k] = piece;
            mean[k] += piece;
        }
    }
    for value in &mut mean {
        *value /= scale;
    }
    for i in 0..n {
        for k in 0..3 {
            gradient[3 * i + k] -= mean[k];
        }
    }
    (value, gradient)
}

fn smooth_radius_penalty(x: ArrayView1<f64>, width: f64, mu: f64) -> (f64, Array1<f64>) {
    let (value, gradient) = smooth_radius(x, width);
    (mu * value, gradient * mu)
}

fn anisotropy_penalty(x: ArrayView1<f64>, nu: f64) -> (f64, Array1<f64>) {
    let (aniso, gradient) = shape_parts(x);
    (nu * aniso, gradient * nu)
}

/// Anisotropy and its gradient at unit strength.
fn shape_parts(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut gradient = Array1::zeros(x.len());
    if n < 2 {
        return (0.0, gradient);
    }
    let mut com = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            com[k] += x[3 * i + k];
        }
    }
    let scale = n as f64;
    for value in &mut com {
        *value /= scale;
    }
    let mut tensor = [[0.0; 3]; 3];
    for i in 0..n {
        let d = [
            x[3 * i] - com[0],
            x[3 * i + 1] - com[1],
            x[3 * i + 2] - com[2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                tensor[a][b] += d[a] * d[b];
            }
        }
    }
    let (evals, evecs) = jacobi3(tensor);
    let (l0, l1, l2) = (evals[0], evals[1], evals[2]);
    let aniso = (l0 - l1).powi(2) + (l1 - l2).powi(2) + (l2 - l0).powi(2);
    let d_aniso = [
        2.0 * (l0 - l1) - 2.0 * (l2 - l0),
        -2.0 * (l0 - l1) + 2.0 * (l1 - l2),
        -2.0 * (l1 - l2) + 2.0 * (l2 - l0),
    ];
    for i in 0..n {
        let d = [
            x[3 * i] - com[0],
            x[3 * i + 1] - com[1],
            x[3 * i + 2] - com[2],
        ];
        for mode in 0..3 {
            let axis = [evecs[0][mode], evecs[1][mode], evecs[2][mode]];
            let proj = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
            let coef = d_aniso[mode] * 2.0 * proj;
            for k in 0..3 {
                gradient[3 * i + k] += coef * axis[k];
            }
        }
    }
    (aniso, gradient)
}

/// Eigenvalues, ascending, and eigenvectors stored by column.
fn jacobi3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..8 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1.0e-15 {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            for r in 0..3 {
                if r == p || r == q {
                    continue;
                }
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
            for r in 0..3 {
                let vip = v[r][p];
                let viq = v[r][q];
                v[r][p] = c * vip - s * viq;
                v[r][q] = s * vip + c * viq;
            }
        }
    }
    let mut order = [0, 1, 2];
    if a[order[0]][order[0]] > a[order[1]][order[1]] {
        order.swap(0, 1);
    }
    if a[order[1]][order[1]] > a[order[2]][order[2]] {
        order.swap(1, 2);
    }
    if a[order[0]][order[0]] > a[order[1]][order[1]] {
        order.swap(0, 1);
    }
    let evals = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let mut evecs = [[0.0; 3]; 3];
    for mode in 0..3 {
        for row in 0..3 {
            evecs[row][mode] = v[row][order[mode]];
        }
    }
    (evals, evecs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d = [0.0; 3];
                let mut r2 = 0.0;
                for k in 0..3 {
                    d[k] = x[3 * i + k] - x[3 * j + k];
                    r2 += d[k] * d[k];
                }
                let inv2 = 1.0 / r2;
                let inv6 = inv2 * inv2 * inv2;
                let inv12 = inv6 * inv6;
                energy += 4.0 * (inv12 - inv6);
                let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
                for k in 0..3 {
                    gradient[3 * i + k] -= coef * d[k];
                    gradient[3 * j + k] += coef * d[k];
                }
            }
        }
        (energy, gradient)
    }

    fn quench(x: ArrayView1<f64>) -> Array1<f64> {
        let mut opt = WarmLbfgs::default();
        opt.minimize(x, 400, |v| Some(lj(v))).1
    }

    fn load_ico() -> Array1<f64> {
        let mut vals = Vec::new();
        for line in include_str!("../../tests/fixtures/lj75_ico.xyz").lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            if let (Ok(x), Ok(y), Ok(z)) = (
                parts[parts.len() - 3].parse::<f64>(),
                parts[parts.len() - 2].parse::<f64>(),
                parts[parts.len() - 1].parse::<f64>(),
            ) {
                vals.extend([x, y, z]);
            }
        }
        assert_eq!(vals.len(), 225);
        Array1::from(vals)
    }

    #[test]
    fn inertia_anisotropy_gradient_matches_a_finite_difference() {
        let mut coords = Vec::new();
        for i in 0..6 {
            let t = i as f64;
            coords.extend([0.2 * t, t.sin(), 0.3 * t - 0.4]);
        }
        let x = Array1::from(coords);
        let (_energy, gradient) = anisotropy_penalty(x.view(), 1.0);
        let step = 1.0e-6;
        for i in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += step;
            minus[i] -= step;
            let e_plus = anisotropy_penalty(plus.view(), 1.0).0;
            let e_minus = anisotropy_penalty(minus.view(), 1.0).0;
            let fd = (e_plus - e_minus) / (2.0 * step);
            assert!(
                (fd - gradient[i]).abs() < 1.0e-4,
                "component {i}: finite difference {fd} gradient {}",
                gradient[i]
            );
        }
    }

    #[test]
    fn inertia_anisotropy_leaves_the_lj75_icosahedron() {
        let raw = load_ico();
        let (start, quenched) = {
            let mut opt = WarmLbfgs::default();
            let (energy, coords, _) = opt.minimize(raw.view(), 800, |v| Some(lj(v)));
            (energy, coords)
        };
        let contact = crate::lattice::nearest_neighbour_scale(quenched.view());
        let best = search(quenched.view(), contact, 1, 1, lj, quench);
        assert!(
            best < -396.282249,
            "compact sphere {best:.6} did not leave the icosahedron {start:.6}"
        );
    }
}
