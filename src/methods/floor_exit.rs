//! Cover, minimum-mode climb, then a plain quench from a compacted relaxation.
//!
//! The climb is the shipped minimum-mode search. The quench minimises a
//! diameter penalty that reads only pair distances of the structure being
//! relaxed, then minimises the plain energy. No target energy is read.

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::methods::activation::{Activation, cover_climb_search};
use crate::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use crate::methods::two_phase::{TwoPhase, penalty};
use crate::methods::warm_lbfgs::WarmLbfgs;

/// Covering displacements, minimum-mode climbs, and plain quenches.
///
/// Returns the lowest plain energy seen. The diameter cutoff is seven tenths
/// of the largest pair distance of the structure entering each relaxation.
pub fn search<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    hops: usize,
    seed: u64,
    mut evaluate: E,
    mut quench: Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64> + Send,
{
    let (start_energy, _) = evaluate(origin);
    let climbed = cover_climb_search(
        origin,
        contact,
        hops,
        seed,
        &mut evaluate,
        &mut quench,
        &Activation::default(),
    );
    let (climbed_energy, _) = evaluate(climbed.view());
    let mut best = start_energy.min(climbed_energy);
    let n = origin.len() / 3;
    if n < 2 || !start_energy.is_finite() {
        return best;
    }
    let mut cfg = Config::recommended(n);
    let two = TwoPhase::relative(0.7, 1.0);
    cfg.two_phase = Some(two);
    let budget = n.saturating_mul(4_000);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut hop_index = 0usize;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        hop_index = hop_index.saturating_add(1);
        let hop = hop_index;
        opt.forget();
        let cutoff = two.cutoff_for(x);
        let (_, phase, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            let (energy, gradient) = evaluate(v);
            let (extra, extra_gradient) = penalty(v, cutoff, two.beta, two.mu);
            Some((energy + extra, gradient + extra_gradient))
        });
        opt.forget();
        let (energy, quenched, _) = opt.minimize(phase.view(), iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(evaluate(v))
        });
        if energy.is_finite() {
            println!(
                "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if energy < best {
                best = energy;
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
    fn cover_climb_and_plain_quench_leaves_the_lj75_icosahedron() {
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
            "plain quench {best:.6} did not leave the icosahedron {start:.6}"
        );
    }
}
