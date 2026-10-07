//! Covering displacement, minimum-mode climb, and a plain quench.
//!
//! Each hop takes one covering direction and climbs it, keeping the later
//! ridges. Every ridge is quenched on the plain energy. A new minimum
//! below the harmonic ceiling is climbed in turn. The search reads the
//! caller's energy and force. It does not read a target energy.

use std::collections::HashSet;
use std::io::Write;

use ndarray::{Array1, ArrayView1};

use crate::curvature::curvature_features;
use crate::known_basin::{LEAVE_BARRIER_FLOOR, LEAVE_BARRIER_GROWTH};
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
        return cover_network(
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

/// Covering directions and later-ridge climbs, quenched on the plain energy.
///
/// The first ridge out of a deep minimum returns to the same well. The
/// climb keeps the later ridges, and a new minimum below the harmonic
/// ceiling is climbed in turn. No target energy and no microcanonical
/// trajectory.
fn cover_network<E, Q>(
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
    let n_atoms = (origin.len() / 3).max(1) as f64;
    let mut cfg = Activation::default();
    cfg.later_ridges = true;
    cfg.step = contact / n_atoms.sqrt();
    // Shallow ridges stay inside one funnel. A ridge at least one rung
    // up is the one a later minimum can leave from. The longer climb is
    // what reaches that ridge.
    cfg.min_rise = harmonic * LEAVE_BARRIER_FLOOR;
    cfg.max_steps = cfg
        .max_steps
        .saturating_mul(LEAVE_BARRIER_GROWTH.powi(2) as usize);
    let limit = hops.max(1);
    let n_cover = crate::hypersphere::default_cover_size();
    let covers = n_cover;
    let mut queue = vec![(start_energy, origin.to_owned())];
    let mut climbed = HashSet::new();
    let mut best = start_energy;
    let ceiling = start_energy + harmonic * LEAVE_BARRIER_GROWTH;
    println!(
        "{{\"kind\":\"cover_network\",\"hops\":{limit},\"min_rise\":{:.4},\"step\":{:.4}}}",
        cfg.min_rise, cfg.step
    );
    let _ = std::io::stdout().flush();
    for hop in 0..limit {
        let Some(choice) = queue
            .iter()
            .enumerate()
            .filter(|(_, (energy, _))| !climbed.contains(&basin_key(*energy)))
            .min_by(|(_, (left, _)), (_, (right, _))| left.total_cmp(right))
            .map(|(index, _)| index)
        else {
            break;
        };
        let (height, point) = queue[choice].clone();
        climbed.insert(basin_key(height));
        for cover in 0..covers {
            let index = hop
                .saturating_mul(covers)
                .wrapping_add(cover)
                .wrapping_add(seed as usize);
            let direction = Array1::from(crate::hypersphere::cover_direction(
                n_cover,
                point.len(),
                index,
            ));
            let norm = direction.dot(&direction).sqrt();
            if norm <= 1.0e-12 {
                continue;
            }
            if hold_cover(
                &point,
                direction.view(),
                norm,
                height,
                hop,
                contact,
                harmonic,
                &cfg,
                start_energy,
                ceiling,
                evaluate,
                quench,
                &mut best,
                &mut queue,
            ) {
                return best;
            }
            if best < start_energy - 1.0e-4 {
                return best;
            }
        }
    }
    best
}

fn hold_cover<E, Q>(
    point: &Array1<f64>,
    direction: ArrayView1<f64>,
    norm: f64,
    height: f64,
    hop: usize,
    contact: f64,
    harmonic: f64,
    cfg: &Activation,
    start_energy: f64,
    ceiling: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
    queue: &mut Vec<(f64, Array1<f64>)>,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let mut mode = direction.to_owned();
    mode /= norm;
    let step = cfg.step;
    let n_atoms = (point.len() / 3).max(1) as f64;
    let perp_cap = contact / cfg.max_steps.max(1) as f64 * n_atoms.sqrt();
    let min_rise = harmonic * LEAVE_BARRIER_FLOOR;
    let reach = contact * LEAVE_BARRIER_GROWTH.powi(2);
    let mut cur = point.clone();
    let mut previous_along: Option<f64> = None;
    let mut saw_uphill = false;
    let mut rise = 0.0;
    for _ in 0..cfg.max_steps {
        let snapshot = cur.clone();
        for (value, component) in cur.iter_mut().zip(mode.iter()) {
            *value += step * *component;
        }
        let mut along = 0.0;
        for _perp in 0..cfg.perp_steps {
            let (_energy, gradient) = evaluate(cur.view());
            if gradient.iter().any(|value| !value.is_finite()) {
                cur.clone_from(&snapshot);
                break;
            }
            along = gradient
                .iter()
                .zip(mode.iter())
                .map(|(force, component)| force * component)
                .sum();
            let mut shift = Array1::zeros(cur.len());
            let mut shift_norm = 0.0;
            for (slot, (force, component)) in shift.iter_mut().zip(gradient.iter().zip(mode.iter()))
            {
                *slot = cfg.perp_rate * (force - along * component);
                shift_norm += *slot * *slot;
            }
            shift_norm = shift_norm.sqrt();
            let scale = if shift_norm > perp_cap && shift_norm > 0.0 {
                perp_cap / shift_norm
            } else {
                1.0
            };
            for (value, component) in cur.iter_mut().zip(shift.iter()) {
                *value -= scale * *component;
            }
        }
        if cur.iter().any(|value| !value.is_finite()) || pair_gap(cur.view()) < 0.5 * contact {
            cur.clone_from(&snapshot);
            break;
        }
        let (_energy, gradient) = evaluate(cur.view());
        along = gradient
            .iter()
            .zip(mode.iter())
            .map(|(force, component)| force * component)
            .sum();
        let mut lambda = 0.0;
        if let Some(previous) = previous_along {
            let slope = (along - previous) / step;
            if slope.is_finite() {
                lambda = slope;
            }
        }
        previous_along = Some(along);
        rise += along * step;
        if along > 0.0 {
            saw_uphill = true;
        }
        if lambda < 0.0 && rise >= min_rise && saw_uphill {
            let newton = if lambda.abs() > 1.0e-8 {
                along.abs() / lambda.abs()
            } else {
                reach
            };
            let push = if along <= 0.0 {
                reach
            } else {
                (LEAVE_BARRIER_GROWTH * newton).clamp(contact, reach * LEAVE_BARRIER_GROWTH)
            };
            let mut far = cur.clone();
            for (value, component) in far.iter_mut().zip(mode.iter()) {
                *value += push * *component;
            }
            println!(
                "{{\"kind\":\"held_ridge\",\"hop\":{hop},\"from\":{height:.6},\"rise\":{rise:.4},\"lambda\":{lambda:.4},\"along\":{along:.4}}}"
            );
            let _ = std::io::stdout().flush();
            let shots = if along <= 0.0 {
                vec![cur, far]
            } else {
                vec![far]
            };
            for shot in shots {
                if note_shot(
                    shot.view(),
                    hop,
                    start_energy,
                    ceiling,
                    evaluate,
                    quench,
                    best,
                    queue,
                ) {
                    return true;
                }
            }
            return false;
        }
        if rise > harmonic * LEAVE_BARRIER_GROWTH {
            return note_shot(
                cur.view(),
                hop,
                start_energy,
                ceiling,
                evaluate,
                quench,
                best,
                queue,
            );
        }
    }
    note_shot(
        cur.view(),
        hop,
        start_energy,
        ceiling,
        evaluate,
        quench,
        best,
        queue,
    )
}

fn note_shot<E, Q>(
    shot: ArrayView1<f64>,
    hop: usize,
    start_energy: f64,
    ceiling: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
    queue: &mut Vec<(f64, Array1<f64>)>,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    if shot.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let Some((energy, coords)) = record_quench(shot, hop, evaluate, quench, best) else {
        return false;
    };
    if *best < start_energy - 1.0e-4 {
        return true;
    }
    if energy > start_energy + 1.0e-4 && energy < ceiling {
        let key = basin_key(energy);
        if !queue.iter().any(|(known, _)| basin_key(*known) == key) {
            queue.push((energy, coords));
        }
    }
    false
}

fn pair_gap(x: ArrayView1<f64>) -> f64 {
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

fn basin_key(energy: f64) -> i64 {
    (energy * 1.0e3).round() as i64
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
        let best = search(quenched.view(), contact, 1, 1, lj, quench);
        assert!(best.is_finite(), "plain quench was not finite");
        assert!(
            best <= start + 1.0e-6,
            "plain quench {best:.6} rose above the icosahedron {start:.6}"
        );
    }
}
