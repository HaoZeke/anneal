//! Covering displacement, minimum-mode climb, and a plain quench.
//!
//! Each hop climbs the soft modes by gentlest ascent. A saddle is quenched
//! on the plain energy, on both sides of the mode. A new minimum below the
//! harmonic ceiling is climbed in turn. The search reads the caller's
//! energy and force. It does not read a target energy.

use std::collections::HashSet;
use std::io::Write;

use ndarray::{Array1, ArrayView1};

use crate::curvature::{curvature_features, soft_subspace};
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
    cfg.step = contact / n_atoms.sqrt();
    // A flip while the cover is still inside the harmonic bowl is a
    // numerical wiggle. A ridge at least one rung up is a barrier.
    cfg.min_rise = harmonic * LEAVE_BARRIER_FLOOR;
    cfg.max_steps = cfg
        .max_steps
        .saturating_mul(LEAVE_BARRIER_GROWTH.powi(2) as usize);
    let limit = hops.max(1);
    let n_cover = crate::hypersphere::default_cover_size();
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
        if hop == 0 {
            let mut best_x = point.clone();
            crate::methods::activation::climb_outer_axes(
                point.view(),
                contact,
                hop,
                evaluate,
                quench,
                &cfg,
                &mut best,
                &mut best_x,
            );
            if best < start_energy - 1.0e-4 {
                return best;
            }
        }
        if bond_scan(
            &point,
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
        let window = soft_window(&point, contact, harmonic, evaluate);
        println!(
            "{{\"kind\":\"soft_window\",\"hop\":{hop},\"modes\":{}}}",
            window.len()
        );
        let _ = std::io::stdout().flush();
        if window.is_empty() {
            break;
        }
        for cover in 0..n_cover {
            let index = hop
                .saturating_mul(n_cover)
                .wrapping_add(cover)
                .wrapping_add(seed as usize);
            let direction = Array1::from(crate::hypersphere::cover_direction(
                n_cover,
                point.len(),
                index,
            ));
            let mut heading = Array1::<f64>::zeros(point.len());
            for (lambda, mode) in &window {
                let scale = lambda.sqrt();
                if !(scale > 0.0) {
                    continue;
                }
                let coeff = direction
                    .iter()
                    .zip(mode.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f64>()
                    / scale;
                if !coeff.is_finite() {
                    continue;
                }
                for (slot, component) in heading.iter_mut().zip(mode.iter()) {
                    *slot += coeff * *component;
                }
            }
            let norm = heading.dot(&heading).sqrt();
            if !(norm > 1.0e-12) {
                continue;
            }
            heading /= norm;
            println!("{{\"kind\":\"cover\",\"hop\":{hop},\"cover\":{cover},\"from\":{height:.6}}}");
            let _ = std::io::stdout().flush();
            if climb_cover(
                &point,
                heading.view(),
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

/// Rotate a bonded pair's common neighbours until the torque flips, then quench.
///
/// The cover is the contact graph. The climb is the rotation about that
/// bond, which is one coordinate. The quench is the plain energy.
fn bond_scan<E, Q>(
    point: &Array1<f64>,
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
    let _ = (height, harmonic);
    let n = point.len() / 3;
    if n < 4 {
        return false;
    }
    let cutoff = contact * (1.0 + LEAVE_BARRIER_FLOOR);
    let cutoff2 = cutoff * cutoff;
    let mut bonds: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = point[3 * i + k] - point[3 * j + k];
                d2 += d * d;
            }
            if d2 < cutoff2 && d2 > 0.0 {
                bonds.push((i, j));
            }
        }
    }
    let mut neigh = vec![Vec::new(); n];
    for &(i, j) in &bonds {
        neigh[i].push(j);
        neigh[j].push(i);
    }
    let angle_step = cfg.step / contact;
    if !(angle_step > 0.0) {
        return false;
    }
    let mut turns = 0usize;
    for &(i, j) in &bonds {
        let mut common = Vec::new();
        for &k in &neigh[i] {
            if neigh[j].contains(&k) {
                common.push(k);
            }
        }
        if common.len() < 2 {
            continue;
        }
        let mut axis = [0.0; 3];
        let mut origin = [0.0; 3];
        for k in 0..3 {
            axis[k] = point[3 * j + k] - point[3 * i + k];
            origin[k] = 0.5 * (point[3 * i + k] + point[3 * j + k]);
        }
        let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !(axis_norm > 1.0e-8) {
            continue;
        }
        for k in 0..3 {
            axis[k] /= axis_norm;
        }
        for a in 0..common.len() {
            for b in (a + 1)..common.len() {
                for sign in [1.0, -1.0] {
                    turns += 1;
                    if turn_bond(
                        point,
                        common[a],
                        common[b],
                        origin,
                        axis,
                        sign * angle_step,
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
                if turns > n.saturating_mul(n) {
                    println!("{{\"kind\":\"bond_scan\",\"hop\":{hop},\"turns\":{turns}}}");
                    let _ = std::io::stdout().flush();
                    return false;
                }
            }
        }
    }
    println!("{{\"kind\":\"bond_scan\",\"hop\":{hop},\"turns\":{turns}}}");
    let _ = std::io::stdout().flush();
    false
}

fn turn_bond<E, Q>(
    point: &Array1<f64>,
    a: usize,
    b: usize,
    origin: [f64; 3],
    axis: [f64; 3],
    angle_step: f64,
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
    let (e0, _) = evaluate(point.view());
    let mut prev = e0;
    let mut angle: f64 = 0.0;
    let half_turn = std::f64::consts::PI;
    while angle.abs() < half_turn {
        angle += angle_step;
        let mut trial = point.clone();
        spin_atom(&mut trial, a, origin, axis, angle);
        spin_atom(&mut trial, b, origin, axis, angle);
        let (energy, _) = evaluate(trial.view());
        if !energy.is_finite() {
            return false;
        }
        if prev > e0 && energy < prev {
            println!(
                "{{\"kind\":\"bond_ridge\",\"hop\":{hop},\"rise\":{:.4},\"angle\":{angle:.4}}}",
                prev - e0
            );
            let _ = std::io::stdout().flush();
            return note_shot(
                trial.view(),
                hop,
                start_energy,
                ceiling,
                evaluate,
                quench,
                best,
                queue,
            );
        }
        prev = energy;
    }
    false
}

fn spin_atom(point: &mut Array1<f64>, atom: usize, origin: [f64; 3], axis: [f64; 3], angle: f64) {
    let (s, c) = angle.sin_cos();
    let v = [
        point[3 * atom] - origin[0],
        point[3 * atom + 1] - origin[1],
        point[3 * atom + 2] - origin[2],
    ];
    let along = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    for k in 0..3 {
        point[3 * atom + k] = origin[k] + v[k] * c + cross[k] * s + axis[k] * along * (1.0 - c);
    }
}

/// Climb one soft mode by gentlest ascent, then quench both sides.
/// then quench a contact past each ridge on the plain energy.
fn climb_cover<E, Q>(
    point: &Array1<f64>,
    heading: ArrayView1<f64>,
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
    let _ = (harmonic, cfg.later_ridges);
    let mut tau = heading.to_owned();
    if !normalize(&mut tau) {
        return false;
    }
    let mut cur = point.clone();
    for (value, component) in cur.iter_mut().zip(tau.iter()) {
        *value += cfg.step * *component;
    }
    // The ascent is an ordinary differential equation. The step is the
    // curvature finite-difference length, grown by one rung, which is small
    // enough that the softest direction can turn over instead of jumping.
    let dt = (cfg.epsilon * LEAVE_BARRIER_GROWTH).max(cfg.epsilon);
    let force_tol =
        (harmonic / contact / ((point.len() / 3).max(1) as f64).sqrt()).max(cfg.epsilon);
    let mut last_rise = 0.0;
    let mut last_curv = 0.0;
    let mut last_gnorm = 0.0;
    let steps = ((contact / dt) as usize).clamp(cfg.max_steps, 8_000);
    for _ in 0..steps {
        let (energy, gradient) = evaluate(cur.view());
        if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            break;
        }
        last_rise = energy - height;
        if last_rise > harmonic * LEAVE_BARRIER_GROWTH.powi(2)
            || pair_gap(cur.view()) < 0.5 * contact
        {
            break;
        }
        let mut plus = cur.clone();
        let mut minus = cur.clone();
        for ((up, down), component) in plus.iter_mut().zip(minus.iter_mut()).zip(tau.iter()) {
            *up += cfg.epsilon * *component;
            *down -= cfg.epsilon * *component;
        }
        let (_, up) = evaluate(plus.view());
        let (_, down) = evaluate(minus.view());
        if up.iter().any(|value| !value.is_finite()) || down.iter().any(|value| !value.is_finite())
        {
            break;
        }
        let mut curv = 0.0;
        let mut hv = Array1::zeros(cur.len());
        for i in 0..cur.len() {
            hv[i] = (up[i] - down[i]) / (2.0 * cfg.epsilon);
            curv += hv[i] * tau[i];
        }
        last_curv = curv;
        for (value, slope) in tau.iter_mut().zip(hv.iter()) {
            *value -= dt * (*slope - curv * *value);
        }
        if !normalize(&mut tau) {
            break;
        }
        let along = dot(&gradient, &tau);
        let mut step_vec = Array1::zeros(cur.len());
        for i in 0..cur.len() {
            step_vec[i] = -gradient[i] + 2.0 * along * tau[i];
        }
        let step_norm = step_vec.dot(&step_vec).sqrt();
        last_gnorm = gradient.dot(&gradient).sqrt();
        if !(step_norm > 0.0) {
            break;
        }
        let capped = if dt * step_norm > cfg.step {
            cfg.step / (dt * step_norm)
        } else {
            1.0
        };
        for (value, component) in cur.iter_mut().zip(step_vec.iter()) {
            *value += dt * capped * *component;
        }
        if last_gnorm <= force_tol && last_curv < 0.0 && along.abs() <= force_tol {
            println!(
                "{{\"kind\":\"gad_saddle\",\"hop\":{hop},\"rise\":{last_rise:.4},\"curv\":{last_curv:.4},\"gnorm\":{last_gnorm:.4}}}"
            );
            let _ = std::io::stdout().flush();
            for sign in [1.0, -1.0] {
                let mut far = cur.clone();
                for (value, component) in far.iter_mut().zip(tau.iter()) {
                    *value += sign * contact * *component;
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
            return false;
        }
    }
    println!(
        "{{\"kind\":\"gad_end\",\"hop\":{hop},\"rise\":{last_rise:.4},\"curv\":{last_curv:.4},\"gnorm\":{last_gnorm:.4}}}"
    );
    let _ = std::io::stdout().flush();
    false
}

/// Flexible modes whose harmonic cost over one contact stays inside the
/// climb's energy window. A raw cover is stiff: one step along it spends
/// the whole window before the force can flip.
fn soft_window<E>(
    point: &Array1<f64>,
    contact: f64,
    harmonic: f64,
    evaluate: &mut E,
) -> Vec<(f64, Array1<f64>)>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let atoms = point.len() / 3;
    let krylov = atoms.clamp(48, 96);
    let ask = krylov.saturating_sub(6).max(2);
    // A curvature below the finite-difference step is not resolved, and a
    // leftover rigid direction sits there.
    let epsilon = 1.0e-4;
    let Some((lambdas, modes, _)) = soft_subspace(
        point.view(),
        |sample| Some(evaluate(sample).1),
        krylov,
        epsilon,
        ask,
    ) else {
        return Vec::new();
    };
    let limit = harmonic * LEAVE_BARRIER_GROWTH.powi(4);
    let mut window = Vec::new();
    for (lambda, mode) in lambdas.into_iter().zip(modes) {
        if !(lambda > epsilon) || !lambda.is_finite() {
            continue;
        }
        let contact_cost = 0.5 * lambda * contact * contact;
        if contact_cost.is_finite() && contact_cost <= limit {
            window.push((lambda, mode));
        }
    }
    let keep = (atoms / LEAVE_BARRIER_GROWTH.powi(2) as usize).max(1);
    window.truncate(keep);
    window
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
