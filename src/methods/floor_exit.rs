//! Covering displacement, minimum-mode climb, and a plain quench.
//!
//! Each hop climbs the soft modes by gentlest ascent. A saddle is quenched
//! on the plain energy, on both sides of the mode. A new minimum below the
//! harmonic ceiling is climbed in turn. The search reads the caller's
//! energy and force. It does not read a target energy.

use std::io::Write;

use ndarray::{Array1, ArrayView1};

use crate::curvature::{curvature_features, project_rigid_with, rigid_basis, soft_subspace};
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
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
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
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send,
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
    // Minima reached by stepping below a converged saddle. The walk climbs
    // those, highest first, instead of a placement that is still in the well.
    let mut bridges: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    seen.insert(basin_key(start_energy));
    let mut best = start_energy;
    let mut walker = origin.to_owned();
    let mut walker_e = start_energy;
    let ceiling = start_energy + harmonic * LEAVE_BARRIER_GROWTH;
    println!(
        "{{\"kind\":\"cover_network\",\"hops\":{limit},\"min_rise\":{:.4},\"step\":{:.4},\"shape\":{:.2}}}",
        cfg.min_rise,
        cfg.step,
        inertia_shape(&walker)
    );
    let _ = std::io::stdout().flush();
    for hop in 0..limit {
        let point = walker.clone();
        let height = walker_e;
        println!(
            "{{\"kind\":\"climb_from\",\"hop\":{hop},\"height\":{height:.6},\"rms\":{:.4}}}",
            separation(origin.view(), &point)
        );
        let _ = std::io::stdout().flush();
        let next = hollow_scan(
            &point,
            height,
            hop,
            contact,
            start_energy,
            ceiling,
            evaluate,
            quench,
            &mut best,
            &mut queue,
        );
        if below_printed_floor(best, start_energy) {
            return best;
        }
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
            if below_printed_floor(best, start_energy) {
                return best;
            }
        }
        if hop == 0
            && bond_scan(
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
            )
        {
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
        // The softest modes, converged to a saddle. A point where the
        // curvature has only just changed sign still falls back into the well.
        // Later hops follow two soft modes. The first hop uses a wider
        // window so the bridge list is not a single valley.
        let n_mode = if hop == 0 {
            (n_cover / LEAVE_BARRIER_GROWTH.powi(3) as usize)
                .clamp(2, 6)
                .min(window.len())
        } else {
            (LEAVE_BARRIER_GROWTH as usize).min(window.len())
        };
        for (_, mode) in window.iter().take(n_mode) {
            for sign in [1.0_f64, -1.0] {
                let mut directed = mode.clone();
                if sign < 0.0 {
                    directed *= sign;
                }
                let norm = directed.dot(&directed).sqrt();
                if !(norm > 1.0e-12) {
                    continue;
                }
                directed /= norm;
                if take_downhill(
                    point.view(),
                    directed.view(),
                    contact,
                    hop,
                    start_energy,
                    ceiling,
                    cfg.min_rise,
                    &cfg,
                    harmonic,
                    evaluate,
                    quench,
                    &mut best,
                    &mut bridges,
                ) {
                    return best;
                }
            }
        }
        // The first hop also follows the soft direction until its curvature
        // changes sign. Later hops spend the budget on converged saddles.
        if hop == 0 {
            for (index, (_, mode)) in window.iter().enumerate().take(n_mode.min(4)) {
                for sign in [1.0_f64, -1.0] {
                    let mut directed = mode.clone();
                    if sign < 0.0 {
                        directed *= sign;
                    }
                    let norm = directed.dot(&directed).sqrt();
                    if !(norm > 1.0e-12) {
                        continue;
                    }
                    directed /= norm;
                    if push_until_negative(
                        &point,
                        &directed,
                        index,
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
                }
            }
        }
        // A cover grown out to the harmonic cap, then climbed from that
        // point. The climb follows the lowest curvature there, which is not
        // the softest mode of the minimum just left.
        let n_grown = if hop == 0 { 2 } else { 1 };
        for cover in 0..n_grown {
            let index = hop
                .saturating_mul(n_cover)
                .wrapping_add(cover)
                .wrapping_add(seed as usize)
                .wrapping_add(n_cover / 2);
            let direction = Array1::from(crate::hypersphere::cover_direction(
                n_cover,
                point.len(),
                index,
            ));
            let grown = grow_cover(
                &point,
                direction.view(),
                height,
                contact,
                harmonic,
                &cfg,
                evaluate,
            );
            println!(
                "{{\"kind\":\"grown\",\"hop\":{hop},\"cover\":{cover},\"rms\":{:.4}}}",
                separation(point.view(), &grown)
            );
            let _ = std::io::stdout().flush();
            if push_until_negative(
                &grown,
                &direction,
                cover,
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
        }
        // A few covers, each walked past its first ridge. One ridge returns
        // to the well it left; the next ridge is the one that can leave.
        let n_use = if hop == 0 {
            (n_cover / LEAVE_BARRIER_GROWTH.powi(4) as usize).clamp(2, 4)
        } else {
            LEAVE_BARRIER_GROWTH as usize
        };
        println!("{{\"kind\":\"cover_chain\",\"hop\":{hop},\"covers\":{n_use}}}");
        let _ = std::io::stdout().flush();
        for cover in 0..n_use {
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
            if take_downhill(
                point.view(),
                heading.view(),
                contact,
                hop,
                start_energy,
                ceiling,
                cfg.min_rise,
                &cfg,
                harmonic,
                evaluate,
                quench,
                &mut best,
                &mut bridges,
            ) {
                return best;
            }
        }
        keep_basins(&queue, f64::INFINITY, start_energy, ceiling, &mut bridges);
        if let Some((energy, coords)) = next_bridge(&bridges, &seen, start_energy, ceiling) {
            println!(
                "{{\"kind\":\"bridge\",\"hop\":{hop},\"energy\":{energy:.6},\"shape\":{:.2},\"rms\":{:.4},\"waiting\":{}}}",
                inertia_shape(&coords),
                separation(origin.view(), &coords),
                bridges.len()
            );
            let _ = std::io::stdout().flush();
            seen.insert(basin_key(energy));
            walker_e = energy;
            walker = coords;
            continue;
        }
        // A distinct neighbour on the shelf is climbed as well. Only the
        // basin just left is skipped.
        let Some((energy, coords)) = next else {
            break;
        };
        if basin_key(energy) == basin_key(height) {
            break;
        }
        walker_e = energy;
        walker = coords;
    }
    best
}

/// Quench both sides of one converged saddle and keep the side below it.
fn take_downhill<E, Q>(
    start: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    contact: f64,
    hop: usize,
    start_energy: f64,
    ceiling: f64,
    min_rise: f64,
    cfg: &Activation,
    harmonic: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
    bridges: &mut Vec<(f64, Array1<f64>)>,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let mut best_x = start.to_owned();
    let (saddle_pose, landed) = crate::methods::activation::connect_cover(
        start,
        mode,
        contact,
        hop,
        evaluate,
        quench,
        best,
        &mut best_x,
    );
    if below_printed_floor(*best, start_energy) {
        return true;
    }
    let Some((saddle_energy, saddle_at, saddle_mode)) = saddle_pose else {
        return false;
    };
    // The first index-1 saddles sit about one rung up and both quenches
    // fall back into the well, or onto the bridge just above the floor.
    // They are not where the climb stops.
    let returning = saddle_energy < start_energy + min_rise;
    if returning {
        let rise = saddle_energy - start_energy;
        println!(
            "{{\"kind\":\"saddle_return\",\"hop\":{hop},\"saddle\":{saddle_energy:.6},\"rise\":{rise:.4}}}"
        );
        let _ = std::io::stdout().flush();
        // The quench on the far side can be a different basin even when the
        // saddle itself is only one rung up. Keep that basin.
        keep_basins(&landed, saddle_energy, start_energy, ceiling, bridges);
        if below_printed_floor(*best, start_energy) {
            return true;
        }
        return step_past_saddle(
            start,
            &saddle_at,
            &saddle_mode,
            contact,
            hop,
            start_energy,
            ceiling,
            harmonic,
            cfg,
            evaluate,
            quench,
            best,
            bridges,
        );
    }
    for (energy, coords) in landed {
        // Above the saddle the push left the adjacent basin.
        if energy + 1.0e-3 >= saddle_energy || !(energy < ceiling) {
            continue;
        }
        if below_printed_floor(energy, start_energy) {
            return true;
        }
        keep_basins(
            &[(energy, coords)],
            saddle_energy,
            start_energy,
            ceiling,
            bridges,
        );
    }
    false
}

/// Leave a returning saddle along its unstable mode and climb from there.
fn step_past_saddle<E, Q>(
    origin: ArrayView1<f64>,
    saddle: &Array1<f64>,
    mode: &Array1<f64>,
    contact: f64,
    hop: usize,
    start_energy: f64,
    ceiling: f64,
    harmonic: f64,
    cfg: &Activation,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
    bridges: &mut Vec<(f64, Array1<f64>)>,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let mut away = 0.0;
    for ((at, from), component) in saddle.iter().zip(origin.iter()).zip(mode.iter()) {
        away += (at - from) * component;
    }
    let sign = if away >= 0.0 { 1.0 } else { -1.0 };
    let mut heading = mode.clone();
    if sign < 0.0 {
        heading *= -1.0;
    }
    let mut past = saddle.clone();
    // A contact already falls back into the well. A fraction of the cover
    // step leaves the saddle without driving two atoms through each other.
    let span = cfg.step * LEAVE_BARRIER_GROWTH;
    for (value, component) in past.iter_mut().zip(heading.iter()) {
        *value += span * component;
    }
    if closest_pair(past.view()) < 0.5 * contact {
        return false;
    }
    let mut queue = Vec::new();
    if note_shot(
        past.view(),
        hop,
        start_energy,
        ceiling,
        evaluate,
        quench,
        best,
        &mut queue,
    ) {
        return true;
    }
    keep_basins(&queue, f64::INFINITY, start_energy, ceiling, bridges);
    push_until_negative(
        &past,
        &heading,
        0,
        start_energy,
        hop,
        contact,
        harmonic,
        cfg,
        start_energy,
        ceiling,
        evaluate,
        quench,
        best,
        bridges,
    )
}

/// The lowest distinct basin still under the harmonic ceiling.
fn next_bridge(
    bridges: &[(f64, Array1<f64>)],
    seen: &std::collections::HashSet<i64>,
    start_energy: f64,
    ceiling: f64,
) -> Option<(f64, Array1<f64>)> {
    bridges
        .iter()
        .filter(|bridge| {
            bridge.0 > start_energy + 1.0e-4
                && bridge.0 < ceiling
                && !seen.contains(&basin_key(bridge.0))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .cloned()
}

/// Keep a quenched basin that is not the minimum the climb left.
fn keep_basins(
    landed: &[(f64, Array1<f64>)],
    saddle_energy: f64,
    start_energy: f64,
    ceiling: f64,
    bridges: &mut Vec<(f64, Array1<f64>)>,
) {
    for (energy, coords) in landed {
        if *energy + 1.0e-3 >= saddle_energy || !(*energy < ceiling) {
            continue;
        }
        if *energy <= start_energy + 1.0e-4 {
            continue;
        }
        let key = basin_key(*energy);
        if bridges.iter().any(|(known, _)| basin_key(*known) == key) {
            continue;
        }
        bridges.push((*energy, coords.clone()));
    }
}

/// Walk a covering direction until the rise meets the cap or two atoms meet.
fn grow_cover<E>(
    point: &Array1<f64>,
    direction: ArrayView1<f64>,
    height: f64,
    contact: f64,
    harmonic: f64,
    cfg: &Activation,
    evaluate: &mut E,
) -> Array1<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let mut scale = cfg.step;
    let mut grown = point.clone();
    let cap = height + harmonic * LEAVE_BARRIER_GROWTH;
    for _ in 0..16 {
        let mut trial = point.clone();
        for (value, component) in trial.iter_mut().zip(direction.iter()) {
            *value += scale * *component;
        }
        if closest_pair(trial.view()) < 0.7 * contact {
            break;
        }
        let (energy, _) = evaluate(trial.view());
        if !energy.is_finite() || energy > cap {
            break;
        }
        grown = trial;
        scale *= 1.35;
    }
    grown
}

/// Six-decimal print of the quenched start. A role quench has left the
/// floor when it is strictly below that print.
fn below_printed_floor(energy: f64, start: f64) -> bool {
    let floor = (start * 1.0e6).round() / 1.0e6;
    energy < floor
}

fn inertia_shape(x: &Array1<f64>) -> f64 {
    let atoms = x.len() / 3;
    if atoms == 0 {
        return 0.0;
    }
    let mut com = [0.0; 3];
    for atom in 0..atoms {
        for axis in 0..3 {
            com[axis] += x[3 * atom + axis];
        }
    }
    let scale = atoms as f64;
    for value in &mut com {
        *value /= scale;
    }
    let mut moment = [[0.0; 3]; 3];
    for atom in 0..atoms {
        let r = [
            x[3 * atom] - com[0],
            x[3 * atom + 1] - com[1],
            x[3 * atom + 2] - com[2],
        ];
        for row in 0..3 {
            for col in 0..3 {
                moment[row][col] += r[row] * r[col];
            }
        }
    }
    let values = jacobi_eigenvalues(moment);
    let low_gap = values[1] - values[0];
    let high_gap = values[2] - values[1];
    if high_gap >= low_gap {
        high_gap
    } else {
        -low_gap
    }
}

/// Eigenvalues of a symmetric 3×3 matrix, ascending.
fn jacobi_eigenvalues(mut moment: [[f64; 3]; 3]) -> [f64; 3] {
    for _ in 0..8 {
        let mut pivot = (0usize, 1usize, moment[0][1].abs());
        if moment[0][2].abs() > pivot.2 {
            pivot = (0, 2, moment[0][2].abs());
        }
        if moment[1][2].abs() > pivot.2 {
            pivot = (1, 2, moment[1][2].abs());
        }
        if pivot.2 < 1.0e-10 {
            break;
        }
        let (p, q) = (pivot.0, pivot.1);
        let app = moment[p][p];
        let aqq = moment[q][q];
        let apq = moment[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let root = (1.0 + tau * tau).sqrt();
        let t = if tau >= 0.0 {
            1.0 / (tau + root)
        } else {
            -1.0 / (-tau + root)
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        moment[p][p] = app - t * apq;
        moment[q][q] = aqq + t * apq;
        moment[p][q] = 0.0;
        moment[q][p] = 0.0;
        for row in 0..3 {
            if row == p || row == q {
                continue;
            }
            let arp = moment[row][p];
            let arq = moment[row][q];
            moment[row][p] = c * arp - s * arq;
            moment[p][row] = moment[row][p];
            moment[row][q] = s * arp + c * arq;
            moment[q][row] = moment[row][q];
        }
    }
    let mut values = [moment[0][0], moment[1][1], moment[2][2]];
    values.sort_by(|left, right| left.total_cmp(right));
    values
}

/// Push a soft covering direction until its curvature is negative, then quench.
///
/// The direction lies in the flexible window, so the rise stays inside the
/// harmonic cap when the curvature turns over. Sideways force is relaxed
/// on each step. Both sides of that point are quenched on the plain energy.
fn push_until_negative<E, Q>(
    point: &Array1<f64>,
    heading: &Array1<f64>,
    cover: usize,
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
    let rise_cap = harmonic * LEAVE_BARRIER_GROWTH;
    let mut cur = point.clone();
    let mut tau = heading.clone();
    let mut last_curv = 0.0;
    let mut last_rise = 0.0;
    let mut negative = false;
    // A fraction of the cover step. A full step walks out of the valley
    // and the curvature stiffens before it can change sign.
    let stride = cfg.step * LEAVE_BARRIER_FLOOR;
    for _step in 0..cfg.max_steps {
        let Some(features) = curvature_features(
            cur.view(),
            |sample| {
                let (_, gradient) = evaluate(sample);
                if gradient.iter().any(|value| !value.is_finite()) {
                    None
                } else {
                    Some(gradient)
                }
            },
            24,
            cfg.epsilon,
        ) else {
            break;
        };
        last_curv = features.lambda_min;
        let along_curv = directional_curvature(&cur, &tau, cfg.epsilon, evaluate).unwrap_or(0.0);
        let turned = (features.lambda_min < 0.0 || along_curv < 0.0) && _step > 0;
        if turned && !negative {
            negative = true;
            let rms = separation(point.view(), &cur);
            println!(
                "{{\"kind\":\"art_negative\",\"hop\":{hop},\"cover\":{cover},\"rise\":{last_rise:.4},\"curv\":{last_curv:.4},\"rms\":{rms:.4}}}"
            );
            let _ = std::io::stdout().flush();
            // The first turnover is a shallow neighbour. A later one,
            // at least one harmonic rise up, is the ridge that is quenched.
            if last_rise >= harmonic {
                for length in [cfg.step, contact] {
                    for sign in [1.0, -1.0] {
                        let mut far = cur.clone();
                        for (value, component) in far.iter_mut().zip(tau.iter()) {
                            *value += sign * length * *component;
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
                if note_shot(
                    cur.view(),
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
        } else if !turned {
            negative = false;
        }
        let mut mode = features.mode;
        if mode.dot(&tau) < 0.0 {
            mode *= -1.0;
        }
        // Stay with the cover when it is still the soft direction.
        // A small overlap means the cover has left the valley.
        if mode.dot(&tau) > 0.5 {
            tau = mode;
        }
        let mut span = stride;
        let snapshot = cur.clone();
        let mut accepted = false;
        for _ in 0..6 {
            let mut trial = snapshot.clone();
            for (value, component) in trial.iter_mut().zip(tau.iter()) {
                *value += span * *component;
            }
            relax_sideways(&mut trial, &tau, evaluate, cfg, contact);
            let (energy, _) = evaluate(trial.view());
            if energy.is_finite()
                && below_printed_floor(energy, start_energy)
                && closest_pair(trial.view()) >= 0.5 * contact
            {
                // The mode has crossed the printed floor. Quench this side.
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
            if energy.is_finite()
                && energy - height <= rise_cap
                && energy + 1.0e-8 >= height
                && closest_pair(trial.view()) >= 0.5 * contact
            {
                cur = trial;
                last_rise = energy - height;
                accepted = true;
                break;
            }
            span *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    println!(
        "{{\"kind\":\"art_end\",\"hop\":{hop},\"cover\":{cover},\"rise\":{last_rise:.4},\"curv\":{last_curv:.4},\"rms\":{:.4}}}",
        separation(point.view(), &cur)
    );
    let _ = std::io::stdout().flush();
    false
}

fn relax_sideways<E>(
    cur: &mut Array1<f64>,
    tau: &Array1<f64>,
    evaluate: &mut E,
    cfg: &Activation,
    contact: f64,
) where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    for _ in 0..(cfg.perp_steps.saturating_mul(2).max(4)) {
        let (energy, gradient) = evaluate(cur.view());
        if !energy.is_finite() {
            return;
        }
        let along = gradient
            .iter()
            .zip(tau.iter())
            .map(|(component, direction)| component * direction)
            .sum::<f64>();
        let mut perp = gradient;
        for (value, component) in perp.iter_mut().zip(tau.iter()) {
            *value -= along * *component;
        }
        let basis = rigid_basis(cur.view());
        project_rigid_with(&mut perp, &basis);
        let perp_norm = perp.dot(&perp).sqrt();
        if !(perp_norm > cfg.epsilon) {
            return;
        }
        let mut span = cfg.perp_rate.min(cfg.step / perp_norm);
        let mut improved = false;
        for _ in 0..6 {
            let mut trial = cur.clone();
            for (value, component) in trial.iter_mut().zip(perp.iter()) {
                *value -= span * *component;
            }
            // Keep the progress already made along the cover.
            let drift = trial
                .iter()
                .zip(cur.iter())
                .zip(tau.iter())
                .map(|((after, before), direction)| (after - before) * direction)
                .sum::<f64>();
            for (value, component) in trial.iter_mut().zip(tau.iter()) {
                *value -= drift * *component;
            }
            let (trial_energy, _) = evaluate(trial.view());
            if trial_energy.is_finite()
                && trial_energy < energy
                && closest_pair(trial.view()) >= 0.5 * contact
            {
                *cur = trial;
                improved = true;
                break;
            }
            span *= 0.5;
        }
        if !improved {
            return;
        }
    }
}

fn directional_curvature<E>(
    point: &Array1<f64>,
    tau: &Array1<f64>,
    epsilon: f64,
    evaluate: &mut E,
) -> Option<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    if !(epsilon > 0.0) {
        return None;
    }
    let mut plus = point.clone();
    let mut minus = point.clone();
    for ((up, down), component) in plus.iter_mut().zip(minus.iter_mut()).zip(tau.iter()) {
        *up += epsilon * *component;
        *down -= epsilon * *component;
    }
    let (_, up) = evaluate(plus.view());
    let (_, down) = evaluate(minus.view());
    if up.iter().any(|value| !value.is_finite()) || down.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let curv = up
        .iter()
        .zip(down.iter())
        .zip(tau.iter())
        .map(|((left, right), component)| (left - right) * component)
        .sum::<f64>()
        / (2.0 * epsilon);
    curv.is_finite().then_some(curv)
}

fn closest_pair(x: ArrayView1<f64>) -> f64 {
    let atoms = x.len() / 3;
    let mut best = f64::MAX;
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            let mut distance2 = 0.0;
            for axis in 0..3 {
                let delta = x[3 * i + axis] - x[3 * j + axis];
                distance2 += delta * delta;
            }
            best = best.min(distance2);
        }
    }
    best.sqrt()
}

/// Move an under-coordinated atom onto an empty face, then quench.
///
/// The faces come from the contact graph. The site is the point that
/// sits one face-edge off that triangle, on either side, when the site
/// is empty. The quench is the plain energy.
fn hollow_scan<E, Q>(
    point: &Array1<f64>,
    height: f64,
    hop: usize,
    contact: f64,
    start_energy: f64,
    ceiling: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
    queue: &mut Vec<(f64, Array1<f64>)>,
) -> Option<(f64, Array1<f64>)>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let n = point.len() / 3;
    if n < 4 || !(contact.is_finite() && contact > 0.0) {
        return None;
    }
    let cutoff2 = (contact * (1.0 + LEAVE_BARRIER_FLOOR)).powi(2);
    let near2 = (contact * LEAVE_BARRIER_GROWTH).powi(2);
    let occupy2 = (contact * LEAVE_BARRIER_FLOOR).powi(2);
    let mut coord = vec![0usize; n];
    let mut neigh = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if pair_distance2(point, i, j) < cutoff2 {
                coord[i] += 1;
                coord[j] += 1;
                neigh[i].push(j);
                neigh[j].push(i);
            }
        }
    }
    let mut order = coord.clone();
    order.sort_unstable();
    let median = order[n / 2];
    let mut faces = Vec::new();
    for i in 0..n {
        for &j in &neigh[i] {
            if j <= i {
                continue;
            }
            for &k in &neigh[i] {
                if k <= j {
                    continue;
                }
                if neigh[j].contains(&k) {
                    faces.push((i, j, k));
                }
            }
        }
    }
    let per_atom = LEAVE_BARRIER_GROWTH as usize;
    let mut placed = 0usize;
    let mut chosen: Option<(f64, Array1<f64>)> = None;
    let here = basin_key(height);
    for atom in 0..n {
        if coord[atom] > median {
            continue;
        }
        let origin = atom_at(point, atom);
        let mut sites = Vec::new();
        for &(a, b, c) in &faces {
            if a == atom || b == atom || c == atom {
                continue;
            }
            let pa = atom_at(point, a);
            let pb = atom_at(point, b);
            let pc = atom_at(point, c);
            let mid = [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ];
            let reach = distance2(origin, mid);
            if reach > near2 {
                continue;
            }
            let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let nnorm =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if nnorm < 1.0e-8 {
                continue;
            }
            let edge = (distance(pa, pb) + distance(pb, pc) + distance(pc, pa)) / 3.0;
            let base2 = distance2(mid, pa);
            let height2 = edge * edge - base2;
            if height2 <= 1.0e-8 {
                continue;
            }
            let height = height2.sqrt();
            for sign in [1.0, -1.0] {
                let site = [
                    mid[0] + sign * height * normal[0] / nnorm,
                    mid[1] + sign * height * normal[1] / nnorm,
                    mid[2] + sign * height * normal[2] / nnorm,
                ];
                let mut occupied = false;
                for other in 0..n {
                    if other == atom {
                        continue;
                    }
                    if distance2(site, atom_at(point, other)) < occupy2 {
                        occupied = true;
                        break;
                    }
                }
                if !occupied {
                    sites.push((reach, site));
                }
            }
        }
        sites.sort_by(|left, right| left.0.total_cmp(&right.0));
        sites.dedup_by(|left, right| distance2(left.1, right.1) < occupy2);
        sites.truncate(per_atom);
        for (_, site) in sites {
            let mut trial = point.clone();
            trial[3 * atom] = site[0];
            trial[3 * atom + 1] = site[1];
            trial[3 * atom + 2] = site[2];
            placed += 1;
            let Some((energy, coords)) = record_quench(trial.view(), hop, evaluate, quench, best)
            else {
                continue;
            };
            if energy > start_energy + 1.0e-4 && energy < ceiling {
                let key = basin_key(energy);
                if !queue.iter().any(|(known, _)| basin_key(*known) == key) {
                    queue.push((energy, coords.clone()));
                }
            }
            if basin_key(energy) != here
                && chosen
                    .as_ref()
                    .map(|(have, _)| energy < *have)
                    .unwrap_or(true)
            {
                chosen = Some((energy, coords));
            }
            if below_printed_floor(*best, start_energy) {
                return chosen;
            }
        }
    }
    println!(
        "{{\"kind\":\"hollow_scan\",\"hop\":{hop},\"faces\":{},\"placed\":{placed}}}",
        faces.len()
    );
    let _ = std::io::stdout().flush();
    chosen
}

fn atom_at(point: &Array1<f64>, atom: usize) -> [f64; 3] {
    [point[3 * atom], point[3 * atom + 1], point[3 * atom + 2]]
}

fn distance2(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    distance2(left, right).sqrt()
}

fn pair_distance2(point: &Array1<f64>, i: usize, j: usize) -> f64 {
    distance2(atom_at(point, i), atom_at(point, j))
}

fn separation(origin: ArrayView1<f64>, point: &Array1<f64>) -> f64 {
    let mut square = 0.0;
    let atoms = (origin.len() / 3).max(1) as f64;
    for (there, here) in origin.iter().zip(point.iter()) {
        let delta = there - here;
        square += delta * delta;
    }
    (square / atoms).sqrt()
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
    // Two bands: the softest, and the next band whose contact cost still
    // fits. The softest band alone returns to the shelf.
    window.truncate(keep.saturating_mul(2));
    window
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
    if below_printed_floor(*best, start_energy) {
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
