//! Recommended search that penalises pairs beyond three nearest neighbours.
//!
//! The cutoff is three times the shortest pair of the structure entering
//! the relaxation, and it is held fixed for that relaxation. The quartic
//! has unit strength and no centroid term. The second step minimises the
//! plain energy. The penalty is not a spherical or inertia term, and it
//! does not build a shell. No target energy is read.

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use crate::methods::two_phase;
use crate::methods::warm_lbfgs::WarmLbfgs;

/// Recommended hops relaxed with the three-contact pair penalty, then a plain quench.
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
    let cfg = Config::recommended(n);
    let budget = n.saturating_mul(4_000);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut hop_index = 0usize;
    let mut best = start_energy;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        hop_index = hop_index.saturating_add(1);
        let hop = hop_index;
        let cutoff = 3.0 * shortest_pair(x);
        opt.forget();
        let (_, bent, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            let (energy, gradient) = evaluate(v);
            let (extra, extra_gradient) = two_phase::penalty(v, cutoff, 1.0, 0.0);
            Some((energy + extra, gradient + extra_gradient))
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
    fn three_contact_pairs_leave_the_lj75_icosahedron() {
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
            "three-contact pairs {best:.6} did not leave the icosahedron {start:.6}"
        );
    }
}
