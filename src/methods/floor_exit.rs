//! Covering displacement, minimum-mode climb, and a plain quench.
//!
//! The softest modes are climbed until the curvature along the mode is
//! negative. A push of a fraction of the cluster's own radius is then
//! quenched on the plain energy. The search reads the caller's energy and
//! force. It does not read a target energy.

use std::io::Write;

use ndarray::{Array1, ArrayView1};

use crate::curvature::{curvature_features, soft_subspace, tracked_mode};
use crate::known_basin::LEAVE_BARRIER_GROWTH;
use crate::methods::activation::Activation;

/// Covering displacements, minimum-mode climbs, and plain quenches.
///
/// Returns the lowest plain energy seen.
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
    if !start_energy.is_finite() || !(contact > 0.0) {
        return start_energy;
    }
    let best = start_energy;
    if let Some(harmonic) = harmonic_contact(origin, contact, &mut evaluate) {
        let growth = LEAVE_BARRIER_GROWTH;
        println!(
            "{{\"kind\":\"harmonic\",\"rise\":{harmonic:.4},\"low\":{:.4},\"high\":{:.4}}}",
            harmonic / growth,
            harmonic * growth
        );
        let _ = std::io::stdout().flush();
        // Covering displacement, minimum-mode climb, plain quench.
        // The microcanonical escape is not this sequence.
        return saddle_push(
            origin,
            contact,
            start_energy,
            harmonic,
            hops,
            seed,
            &mut evaluate,
            &mut quench,
        )
        .min(best);
    }
    best
}

/// Climb the softest modes to a negative curvature, then quench a push
/// whose root-mean-square length is a fraction of the cluster radius.
///
/// The launch direction is one vector of the covering, so the climb is
/// not free to fall back along the mode it just left. No target energy
/// and no microcanonical trajectory.
fn saddle_push<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    hops: usize,
    seed: u64,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    if hops == 0 {
        return start_energy;
    }
    let modes_wanted = hops.clamp(1, 24);
    let n_atoms = (origin.len() / 3).max(1) as f64;
    let push_rms = (spread(origin) / LEAVE_BARRIER_GROWTH.powi(2)).max(contact / n_atoms.sqrt());
    let push = push_rms * n_atoms.sqrt();
    let Some((lambdas, modes, _)) = soft_subspace(
        origin,
        |sample| Some(evaluate(sample).1),
        modes_wanted.saturating_mul(2).max(12),
        1.0e-4,
        modes_wanted,
    ) else {
        return start_energy;
    };
    let mut best = start_energy;
    let n_cover = crate::hypersphere::default_cover_size();
    let cap = contact / LEAVE_BARRIER_GROWTH.powi(2);
    println!("{{\"kind\":\"saddle_push\",\"modes\":{modes_wanted},\"push_rms\":{push_rms:.4}}}");
    let _ = std::io::stdout().flush();
    for (index, mode0) in modes.into_iter().enumerate() {
        let cover = Array1::from(crate::hypersphere::cover_direction(
            n_cover,
            origin.len(),
            index.wrapping_add(seed as usize),
        ));
        let mut mode = mode0;
        if cover.dot(&mode) < 0.0 {
            mode *= -1.0;
        }
        for sign in [1.0_f64, -1.0] {
            let mut here = origin.to_owned();
            let lambda0 = lambdas.get(index).copied().unwrap_or(1.0).abs().max(1.0);
            let launch = (harmonic / (2.0 * lambda0))
                .sqrt()
                .clamp(contact / LEAVE_BARRIER_GROWTH.powi(4), cap);
            for (value, component) in here.iter_mut().zip(mode.iter()) {
                *value += sign * launch * *component;
            }
            let mut previous = mode.clone();
            if sign < 0.0 {
                previous *= -1.0;
            }
            let mut pushed = false;
            for _step in 0..Activation::default().max_steps {
                let Some((lambda, tracked)) =
                    tracked_mode(here.view(), previous.view(), 8, 1.0e-4, |sample| {
                        Some(evaluate(sample).1)
                    })
                else {
                    break;
                };
                previous = tracked;
                let (energy, gradient) = evaluate(here.view());
                if !energy.is_finite() {
                    break;
                }
                let parallel: f64 = gradient
                    .iter()
                    .zip(previous.iter())
                    .map(|(force, component)| force * component)
                    .sum();
                if lambda < 0.0 && parallel.abs() < 0.2 {
                    for way in [1.0_f64, -1.0] {
                        let mut landed = here.clone();
                        for (value, component) in landed.iter_mut().zip(previous.iter()) {
                            *value += way * push * *component;
                        }
                        if let Some((quench_energy, _)) =
                            record_quench(landed.view(), index, evaluate, quench, &mut best)
                            && quench_energy < start_energy - 1.0e-4
                        {
                            return best;
                        }
                    }
                    pushed = true;
                    break;
                }
                if energy - start_energy > harmonic * LEAVE_BARRIER_GROWTH {
                    break;
                }
                let parallel_step = if lambda.abs() < 1.0e-8 {
                    0.0
                } else {
                    (parallel / lambda.abs()).clamp(-cap, cap)
                };
                for (value, component) in here.iter_mut().zip(previous.iter()) {
                    *value += parallel_step * *component;
                }
                if here.iter().any(|value| !value.is_finite())
                    || closest_pair(&here) < 0.5 * contact
                {
                    break;
                }
            }
            if !pushed
                && let Some((quench_energy, _)) =
                    record_quench(here.view(), index, evaluate, quench, &mut best)
                && quench_energy < start_energy - 1.0e-4
            {
                return best;
            }
            if best < start_energy - 1.0e-4 {
                return best;
            }
        }
    }
    best
}
fn harmonic_contact<E>(origin: ArrayView1<f64>, contact: f64, evaluate: &mut E) -> Option<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let features = curvature_features(origin, |point| Some(evaluate(point).1), 24, 1.0e-4)?;
    if !features.lambda_min.is_finite() || features.lambda_min <= 0.0 {
        return None;
    }
    let harmonic = 0.5 * features.lambda_min * contact * contact;
    harmonic.is_finite().then_some(harmonic)
}

/// Root-mean-square distance from the centre of mass.
fn spread(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n == 0 {
        return 0.0;
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
    let mut acc = 0.0;
    for i in 0..n {
        for k in 0..3 {
            let delta = x[3 * i + k] - com[k];
            acc += delta * delta;
        }
    }
    (acc / scale).sqrt()
}
fn record_quench<E, Q>(
    point: ArrayView1<f64>,
    hop: usize,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
) -> Option<(f64, Array1<f64>)>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let quenched = quench(point);
    let (energy, _) = evaluate(quenched.view());
    if !energy.is_finite() {
        return None;
    }
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
    );
    if energy < *best {
        *best = energy;
    }
    Some((energy, quenched))
}

fn closest_pair(x: &Array1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut best = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance2 = 0.0;
            for k in 0..3 {
                let delta = x[3 * i + k] - x[3 * j + k];
                distance2 += delta * delta;
            }
            best = best.min(distance2);
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
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        opt.minimize(x, 600, |v| Some(lj(v))).1
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
    fn cover_climb_and_plain_quench_from_the_lj75_icosahedron() {
        let raw = load_ico();
        let (start, quenched) = {
            let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
            let (energy, coords, _) = opt.minimize(raw.view(), 800, |v| Some(lj(v)));
            (energy, coords)
        };
        let contact = crate::lattice::nearest_neighbour_scale(quenched.view());
        let best = search(quenched.view(), contact, 16, 1, lj, quench);
        assert!(best.is_finite(), "plain quench was not finite");
        assert!(
            best <= start + 1.0e-6,
            "plain quench {best:.6} rose above the icosahedron {start:.6}"
        );
    }
}
