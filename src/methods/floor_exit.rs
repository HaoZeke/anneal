//! Covering displacement, minimum-mode climb, and a plain quench.
//!
//! Each hop takes one covering direction and climbs it, keeping the later
//! ridges. Every ridge is quenched on the plain energy. A new minimum
//! below the harmonic ceiling is climbed in turn. The search reads the
//! caller's energy and force. It does not read a target energy.

use std::collections::HashSet;
use std::io::Write;

use ndarray::{Array1, ArrayView1};

use crate::curvature::{curvature_features, project_rigid_with, rigid_basis};
use crate::known_basin::{LEAVE_BARRIER_FLOOR, LEAVE_BARRIER_GROWTH, LEAVE_RUNGS};
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
    cfg.step = contact / n_atoms.sqrt();
    // A flip while the cover is still inside the harmonic bowl is a
    // numerical wiggle. A ridge at least one rung up is a barrier.
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
            println!("{{\"kind\":\"cover\",\"hop\":{hop},\"cover\":{cover},\"from\":{height:.6}}}");
            let _ = std::io::stdout().flush();
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
    // The cover is held until the force along it changes sign. Negative
    // curvature alone is still the uphill side of the ridge, and a quench
    // from there falls back into the well the climb left.
    let mut heading = direction.to_owned();
    heading /= norm;
    let basis = rigid_basis(point.view());
    project_rigid_with(&mut heading, &basis);
    if !normalize(&mut heading) {
        return false;
    }
    let step = cfg.step;
    let mut cur = point.clone();
    let mut previous_along: Option<f64> = None;
    let mut saw_uphill = false;
    let mut ridges = 0usize;
    let mut last_rise = 0.0;
    for _ in 0..cfg.max_steps {
        if !step_along(&mut cur, &heading, step, contact, cfg.epsilon, evaluate) {
            break;
        }
        let (energy, gradient) = evaluate(cur.view());
        if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            break;
        }
        let along = dot(&gradient, &heading);
        last_rise = energy - height;
        let lambda = directional_curvature(&cur, &heading, cfg.epsilon, evaluate).unwrap_or(0.0);
        if along > 0.0 {
            saw_uphill = true;
        }
        let flip =
            saw_uphill && previous_along.is_some_and(|previous| previous > 0.0 && along <= 0.0);
        if flip && lambda < 0.0 && last_rise >= cfg.min_rise {
            println!(
                "{{\"kind\":\"held_ridge\",\"hop\":{hop},\"from\":{height:.6},\"rise\":{last_rise:.4},\"lambda\":{lambda:.4},\"along\":{along:.4}}}"
            );
            let _ = std::io::stdout().flush();
            if push_lowest_mode(
                &cur,
                &heading,
                hop,
                contact,
                step,
                cfg,
                start_energy,
                ceiling,
                evaluate,
                quench,
                best,
                queue,
            ) {
                return true;
            }
            ridges += 1;
            saw_uphill = false;
            if ridges >= LEAVE_RUNGS {
                return false;
            }
        }
        if last_rise > harmonic * LEAVE_BARRIER_GROWTH {
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
        previous_along = Some(along);
    }
    println!(
        "{{\"kind\":\"cover_end\",\"hop\":{hop},\"ridges\":{ridges},\"rise\":{last_rise:.4}}}"
    );
    let _ = std::io::stdout().flush();
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

/// One step along `heading`, then a slide downhill in the perpendicular
/// plane. The heading itself stays put: it is the cover, not the softest
/// well of the minimum.
fn step_along<E>(
    cur: &mut Array1<f64>,
    heading: &Array1<f64>,
    step: f64,
    contact: f64,
    epsilon: f64,
    evaluate: &mut E,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let snapshot = cur.clone();
    for (value, component) in cur.iter_mut().zip(heading.iter()) {
        *value += step * *component;
    }
    if !cur.iter().all(|value| value.is_finite()) || pair_gap(cur.view()) < 0.5 * contact {
        cur.clone_from(&snapshot);
        return false;
    }
    let budget = ((cur.len() / 3).max(1) as f64).sqrt().round().max(1.0) as usize;
    for _ in 0..budget {
        let (energy, gradient) = evaluate(cur.view());
        if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            cur.clone_from(&snapshot);
            return false;
        }
        let along = dot(&gradient, heading);
        let mut perp = gradient;
        for (value, component) in perp.iter_mut().zip(heading.iter()) {
            *value -= along * *component;
        }
        let basis = rigid_basis(cur.view());
        project_rigid_with(&mut perp, &basis);
        let perp_norm = perp.dot(&perp).sqrt();
        if !(perp_norm > along.abs()) || perp_norm <= epsilon {
            break;
        }
        let mut span = step;
        let mut accepted = false;
        while span > epsilon {
            let mut trial = cur.clone();
            for (value, component) in trial.iter_mut().zip(perp.iter()) {
                *value -= span * (*component / perp_norm);
            }
            if trial.iter().all(|value| value.is_finite())
                && pair_gap(trial.view()) >= 0.5 * contact
            {
                let (trial_energy, _) = evaluate(trial.view());
                if trial_energy.is_finite() && trial_energy <= energy {
                    cur.clone_from(&trial);
                    accepted = true;
                    break;
                }
            }
            span *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    true
}

/// Quench both sides of the lowest mode at the ridge.
///
/// The cover that arrived here is not that mode. Lengths are the climb
/// step and the contact distance, grown by the same rung ratio the climb
/// uses for its ceiling.
fn push_lowest_mode<E, Q>(
    cur: &Array1<f64>,
    heading: &Array1<f64>,
    hop: usize,
    contact: f64,
    step: f64,
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
    let mut mode = heading.clone();
    if let Some(features) = curvature_features(
        cur.view(),
        |point| Some(evaluate(point).1),
        cfg.lanczos_steps.saturating_mul(2),
        cfg.epsilon,
    ) {
        if features.lambda_min < 0.0 && features.lambda_min.is_finite() {
            mode = features.mode;
            if dot(&mode, heading) < 0.0 {
                for value in mode.iter_mut() {
                    *value = -*value;
                }
            }
        }
    }
    if !normalize(&mut mode) {
        return false;
    }
    let lengths = [
        step,
        contact,
        contact * LEAVE_BARRIER_GROWTH,
        contact * LEAVE_BARRIER_GROWTH.powi(2),
    ];
    for sign in [1.0, -1.0] {
        for length in lengths {
            let mut far = cur.clone();
            for (value, component) in far.iter_mut().zip(mode.iter()) {
                *value += sign * length * *component;
            }
            if pair_gap(far.view()) < 0.5 * contact {
                continue;
            }
            if note_shot(
                far.view(),
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
    }
    false
}

fn directional_curvature<E>(
    cur: &Array1<f64>,
    heading: &Array1<f64>,
    epsilon: f64,
    evaluate: &mut E,
) -> Option<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    if !(epsilon > 0.0) {
        return None;
    }
    let mut plus = cur.clone();
    let mut minus = cur.clone();
    for ((up, down), component) in plus.iter_mut().zip(minus.iter_mut()).zip(heading.iter()) {
        *up += epsilon * *component;
        *down -= epsilon * *component;
    }
    let (_, up) = evaluate(plus.view());
    let (_, down) = evaluate(minus.view());
    if up.iter().any(|value| !value.is_finite()) || down.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let slope = up
        .iter()
        .zip(down.iter())
        .zip(heading.iter())
        .map(|((left, right), component)| (left - right) * component)
        .sum::<f64>();
    let lambda = slope / (2.0 * epsilon);
    lambda.is_finite().then_some(lambda)
}

fn dot(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn normalize(vector: &mut Array1<f64>) -> bool {
    let norm = vector.dot(vector).sqrt();
    if !(norm > 1.0e-12) {
        return false;
    }
    *vector /= norm;
    true
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
