//! Covering displacement, minimum-mode climb, and a plain quench.
//!
//! Each hop climbs one covering direction and quenches that point on the
//! plain energy. The same direction seeds a microcanonical escape. A
//! modified-dimer softening bends the launch toward the lowest mode, and
//! the kinetic energy starts at the harmonic cost of one contact step on
//! the soft curvature of the entering structure. The quench of the escape
//! is kept when its rise sits under the adaptive threshold, so a chain can
//! cross a funnel by a series of uphill minima. The kick is not allowed to
//! fall below that harmonic cost. The search reads the caller's energy and
//! force. It does not read a target energy.

use std::collections::HashSet;
use std::io::Write;

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::curvature::{curvature_features, soft_subspace, tracked_mode};
use crate::known_basin::{LEAVE_BARRIER_FLOOR, LEAVE_BARRIER_GROWTH};
use crate::methods::activation::{Activation, cover_climb_quench, cover_climb_search};
use crate::methods::minima_hopping::{
    EscapeFeedback, MdEscapeConfig, MdEscapeGeometry, Visit, nve_escape_seeded,
};

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
    let mut best = start_energy;
    if let Some(harmonic) = harmonic_contact(origin, contact, &mut evaluate) {
        let growth = LEAVE_BARRIER_GROWTH;
        println!(
            "{{\"kind\":\"harmonic\",\"rise\":{harmonic:.4},\"low\":{:.4},\"high\":{:.4}}}",
            harmonic / growth,
            harmonic * growth
        );
        let _ = std::io::stdout().flush();
        best = best.min(equilateral_cycles(
            origin,
            contact,
            start_energy,
            &mut evaluate,
            &mut quench,
        ));
        if best < start_energy - 1.0e-4 {
            return best;
        }
        if hops >= 32 {
            best = best.min(softened_chain(
                origin,
                contact,
                start_energy,
                harmonic,
                hops,
                seed,
                &mut evaluate,
                &mut quench,
            ));
            return best;
        }
        if hops >= 8 {
            best = best.min(band_search(
                origin,
                contact,
                start_energy,
                harmonic,
                &mut evaluate,
                &mut quench,
            ));
            return best;
        }
        if hops >= 4 {
            best = best.min(band_search(
                origin,
                contact,
                start_energy,
                harmonic,
                &mut evaluate,
                &mut quench,
            ));
            if best < start_energy - 1.0e-4 {
                return best;
            }
        }
        best = best.min(subspace_flight(
            origin,
            contact,
            start_energy,
            harmonic,
            hops,
            seed,
            &mut evaluate,
            &mut quench,
        ));
        if best < start_energy - 1.0e-4 {
            return best;
        }
        best = best.min(contact_graph(
            origin,
            contact,
            start_energy,
            harmonic,
            &mut evaluate,
            &mut quench,
        ));
        if best < start_energy - 1.0e-4 {
            return best;
        }
        best = best.min(mix_subspace(
            origin,
            contact,
            start_energy,
            harmonic,
            &mut evaluate,
            &mut quench,
        ));
        if best < start_energy - 1.0e-4 {
            return best;
        }
        best = best.min(follow_modes(
            origin,
            contact,
            start_energy,
            harmonic,
            hops,
            &mut evaluate,
            &mut quench,
        ));
        if best < start_energy - 1.0e-4 {
            return best;
        }
        // The centred cover and the uphill chain are the short-budget exit.
        if hops > 0 && hops <= 4 {
            best = best.min(cover_ridges(
                origin,
                contact,
                start_energy,
                harmonic,
                hops,
                &mut evaluate,
                &mut quench,
            ));
            if best < start_energy - 1.0e-4 {
                return best;
            }
            best = best.min(hop_chain(
                origin,
                contact,
                start_energy,
                harmonic,
                hops,
                seed,
                &mut evaluate,
                &mut quench,
            ));
            if best < start_energy - 1.0e-4 {
                return best;
            }
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
            if climbed_energy.is_finite() {
                best = best.min(climbed_energy);
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

/// Energy bin of a quenched minimum. Distinct minima of this cluster sit
/// well above a thousandth of the well depth.
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

fn basin_key(energy: f64) -> usize {
    let milli = (energy * 1.0e3).round() as i64;
    milli as u64 as usize
}

fn visit_name(visit: Visit) -> &'static str {
    match visit {
        Visit::Same => "same",
        Visit::Known => "known",
        Visit::New => "new",
    }
}

/// Repeated descent of the neighbour-angle mismatch, then a plain quench.
///
/// Neighbours are pairs inside the contact shell. The target cosine is
/// one half, the angle of an equilateral triangle of contacts. The plain
/// energy is quenched after each spell of that descent.
fn equilateral_cycles<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let cutoff = contact * (1.0 + LEAVE_BARRIER_FLOOR);
    let step = contact / LEAVE_BARRIER_GROWTH.powi(2);
    let mut best = start_energy;
    let covers = 8usize;
    println!("{{\"kind\":\"equilateral\",\"cutoff\":{cutoff:.4},\"step\":{step:.4}}}");
    let _ = std::io::stdout().flush();
    for index in 0..covers {
        let raw = crate::hypersphere::cover_direction(covers, origin.len(), index);
        let mut direction = Array1::from(raw);
        let norm = direction.dot(&direction).sqrt();
        if norm < 1.0e-12 {
            continue;
        }
        direction /= norm;
        let mut here = origin.to_owned();
        // The symmetric icosahedron is a critical point of the angle
        // mismatch. A covering displacement makes the gradient nonzero.
        for (value, component) in here.iter_mut().zip(direction.iter()) {
            *value += (contact / LEAVE_BARRIER_GROWTH) * *component;
        }
        for cycle in 0..4 {
            for _ in 0..16 {
                let gradient = equilateral_grad(here.view(), cutoff);
                let norm = gradient.dot(&gradient).sqrt();
                if norm < 1.0e-8 {
                    break;
                }
                for (value, component) in here.iter_mut().zip(gradient.iter()) {
                    *value -= step * *component / norm;
                }
                if here.iter().any(|value| !value.is_finite())
                    || closest_pair(&here) < 0.5 * contact
                {
                    break;
                }
            }
            let quenched = quench(here.view());
            let (energy, _) = evaluate(quenched.view());
            if !energy.is_finite() {
                break;
            }
            println!(
                "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{index},\"role\":\"quench\"}}"
            );
            println!(
                "{{\"kind\":\"equilateral_cycle\",\"cover\":{index},\"cycle\":{cycle},\"energy\":{energy:.6}}}"
            );
            let _ = std::io::stdout().flush();
            if energy < best {
                best = energy;
            }
            if best < start_energy - 1.0e-4 {
                return best;
            }
            here = quenched;
        }
    }
    best
}

fn equilateral_grad(x: ArrayView1<f64>, cutoff: f64) -> Array1<f64> {
    let epsilon = 1.0e-4;
    let base = equilateral_mismatch(x, cutoff);
    let mut gradient = Array1::<f64>::zeros(x.len());
    for i in 0..x.len() {
        let mut shifted = x.to_owned();
        shifted[i] += epsilon;
        gradient[i] = (equilateral_mismatch(shifted.view(), cutoff) - base) / epsilon;
    }
    gradient
}

fn equilateral_mismatch(x: ArrayView1<f64>, cutoff: f64) -> f64 {
    let n = x.len() / 3;
    let mut neighbours = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance2 = 0.0;
            for k in 0..3 {
                let delta = x[3 * i + k] - x[3 * j + k];
                distance2 += delta * delta;
            }
            let distance = distance2.sqrt();
            if distance < cutoff && distance > 1.0e-8 {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
    }
    let mut mismatch = 0.0;
    for i in 0..n {
        let shell = &neighbours[i];
        for a in 0..shell.len() {
            for b in (a + 1)..shell.len() {
                let ja = shell[a];
                let jb = shell[b];
                let mut left = [0.0; 3];
                let mut right = [0.0; 3];
                for k in 0..3 {
                    left[k] = x[3 * ja + k] - x[3 * i + k];
                    right[k] = x[3 * jb + k] - x[3 * i + k];
                }
                let left_norm = left.iter().map(|v| v * v).sum::<f64>().sqrt();
                let right_norm = right.iter().map(|v| v * v).sum::<f64>().sqrt();
                if left_norm < 1.0e-8 || right_norm < 1.0e-8 {
                    continue;
                }
                let cosine = left
                    .iter()
                    .zip(right.iter())
                    .map(|(u, v)| u * v)
                    .sum::<f64>()
                    / (left_norm * right_norm);
                let delta = cosine - 0.5;
                mismatch += delta * delta;
            }
        }
    }
    mismatch
}

/// Microcanonical flight in the soft subspace, then a plain quench.
///
/// The launch is one covering direction of the lowest modes, with enough
/// kinetic energy for each of them to reach a harmonic contact. The
/// trajectory stops when the root-mean-square displacement matches that
/// simultaneous step.
fn subspace_flight<E, Q>(
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
    let count = 12usize;
    let n_atoms = (origin.len() / 3).max(1) as f64;
    let max_rms = contact / n_atoms.sqrt() * (count as f64).sqrt();
    let kinetic = harmonic * count as f64;
    let escape = MdEscapeConfig {
        geometry: MdEscapeGeometry::RigidQuotient,
        minimum_rise: harmonic,
        softening: None,
        potential_minima: usize::MAX,
        maximum_steps: MdEscapeConfig::default().maximum_steps * 2,
        max_rms,
        min_well_rms: 0.0,
        ..MdEscapeConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best = start_energy;
    let mut walker = origin.to_owned();
    let mut walker_energy = start_energy;
    let trials = hops.max(1).min(400);
    println!(
        "{{\"kind\":\"subspace\",\"modes\":{count},\"trials\":{trials},\"kinetic\":{kinetic:.4},\"max_rms\":{max_rms:.4}}}"
    );
    let _ = std::io::stdout().flush();
    for index in 0..trials {
        let Some((_, modes, _)) = soft_subspace(
            walker.view(),
            |point| Some(evaluate(point).1),
            36,
            1.0e-4,
            count,
        ) else {
            break;
        };
        let coeff =
            crate::hypersphere::cover_direction(trials.max(modes.len()), modes.len(), index);
        let mut scale = 0.0;
        for value in &coeff {
            scale += value * value;
        }
        scale = scale.sqrt();
        if scale < 1.0e-12 {
            continue;
        }
        let mut direction = Array1::<f64>::zeros(walker.len());
        for (j, mode) in modes.iter().enumerate() {
            let weight = coeff.get(j).copied().unwrap_or(0.0) / scale;
            for (value, component) in direction.iter_mut().zip(mode.iter()) {
                *value += weight * *component;
            }
        }
        let mut surface = |point: ArrayView1<f64>| {
            let (energy, gradient) = evaluate(point);
            if energy.is_finite() && gradient.iter().all(|value| value.is_finite()) {
                Some((energy, gradient))
            } else {
                None
            }
        };
        let Ok(report) = nve_escape_seeded(
            walker.view(),
            kinetic,
            Some(direction.view()),
            &escape,
            &mut surface,
            &mut rng,
        ) else {
            println!("{{\"kind\":\"subspace_escape\",\"hop\":{index},\"ok\":false}}");
            continue;
        };
        let quenched = quench(report.far_position.view());
        let (energy, _) = evaluate(quenched.view());
        if !energy.is_finite() {
            continue;
        }
        println!(
            "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{index},\"role\":\"quench\"}}"
        );
        println!(
            "{{\"kind\":\"subspace_escape\",\"hop\":{index},\"steps\":{},\"far_rms\":{:.4},\"walker\":{walker_energy:.6},\"energy\":{energy:.6}}}",
            report.steps, report.far_rms
        );
        let _ = std::io::stdout().flush();
        if energy < best {
            best = energy;
        }
        if best < start_energy - 1.0e-4 {
            return best;
        }
        // A landing within one harmonic contact is the next place to leave
        // from. A return to the same well is not.
        if energy < walker_energy + harmonic && (energy - walker_energy).abs() > 1.0e-3 {
            walker = quenched;
            walker_energy = energy;
        }
    }
    best
}

/// Displacements along the non-trivial modes of the contact graph.
///
/// Each atom moves radially by its mode weight. The contact cutoff and the
/// step lengths are the neighbour distance of the entering structure.
fn contact_graph<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let n = origin.len() / 3;
    if n < 4 {
        return start_energy;
    }
    let cutoff = contact * LEAVE_BARRIER_GROWTH;
    let mut laplacian = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance2 = 0.0;
            for k in 0..3 {
                let delta = origin[3 * i + k] - origin[3 * j + k];
                distance2 += delta * delta;
            }
            let distance = distance2.sqrt();
            if distance < cutoff {
                laplacian[[i, i]] += 1.0;
                laplacian[[j, j]] += 1.0;
                laplacian[[i, j]] -= 1.0;
                laplacian[[j, i]] -= 1.0;
            }
        }
    }
    let (values, vectors) = crate::spectral::symmetric_eigen(laplacian.view(), 48);
    let mut com = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            com[k] += origin[3 * i + k];
        }
    }
    for value in &mut com {
        *value /= n as f64;
    }
    let mut best = start_energy;
    let mut landed = Vec::new();
    let mut seen = HashSet::new();
    let ceiling = start_energy + harmonic * LEAVE_BARRIER_GROWTH.powi(2);
    let amplitudes = [
        contact * LEAVE_BARRIER_GROWTH,
        contact * LEAVE_BARRIER_GROWTH.powi(2),
        contact * (LEAVE_BARRIER_GROWTH.powi(2) + LEAVE_BARRIER_GROWTH),
    ];
    let mut used = 0usize;
    for mode_index in 0..n {
        if values[mode_index].abs() < 1.0e-6 {
            continue;
        }
        if used >= 6 {
            break;
        }
        used += 1;
        let mut weight = Array1::<f64>::zeros(n);
        for atom in 0..n {
            weight[atom] = vectors[[atom, mode_index]];
        }
        let scale = weight.dot(&weight).sqrt();
        if scale < 1.0e-12 {
            continue;
        }
        weight /= scale;
        for sign in [1.0_f64, -1.0] {
            for amplitude in amplitudes {
                let mut here = origin.to_owned();
                for atom in 0..n {
                    let mut radial = [0.0; 3];
                    let mut radial_norm = 0.0;
                    for k in 0..3 {
                        radial[k] = origin[3 * atom + k] - com[k];
                        radial_norm += radial[k] * radial[k];
                    }
                    radial_norm = radial_norm.sqrt();
                    if radial_norm < 1.0e-8 {
                        continue;
                    }
                    let step = sign * amplitude * weight[atom] / radial_norm;
                    for k in 0..3 {
                        here[3 * atom + k] += step * radial[k];
                    }
                }
                if here.iter().any(|value| !value.is_finite())
                    || closest_pair(&here) < 0.5 * contact
                {
                    continue;
                }
                let quenched = quench(here.view());
                if keep(
                    quenched.view(),
                    mode_index,
                    start_energy,
                    ceiling,
                    evaluate,
                    &mut best,
                    &mut landed,
                    &mut seen,
                ) {
                    return best;
                }
            }
        }
    }
    println!("{{\"kind\":\"graph\",\"modes\":{used},\"best\":{best:.6}}}");
    let _ = std::io::stdout().flush();
    best
}

/// Combinations of the softest modes, at one and two contact lengths.
///
/// A pure mode can stay stable while a mixture at the same length does
/// not. The lowest curvature of each mixture is quenched when it is negative.
fn mix_subspace<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let count = 16usize;
    let Some((_, modes, _)) =
        soft_subspace(origin, |point| Some(evaluate(point).1), 48, 1.0e-4, count)
    else {
        return start_energy;
    };
    let mut best = start_energy;
    let mut landed = Vec::new();
    let mut seen = HashSet::new();
    let ceiling = start_energy + harmonic * LEAVE_BARRIER_GROWTH.powi(2);
    let trials = 32usize;
    let mut softest = f64::MAX;
    let mut soft_rise = 0.0;
    let mut soft_point: Option<Array1<f64>> = None;
    let mut soft_mode: Option<Array1<f64>> = None;
    for index in 0..trials {
        let coeff = crate::hypersphere::cover_direction(trials, modes.len(), index);
        let mut direction = Array1::<f64>::zeros(origin.len());
        for (j, mode) in modes.iter().enumerate() {
            let weight = coeff.get(j).copied().unwrap_or(0.0);
            for (value, component) in direction.iter_mut().zip(mode.iter()) {
                *value += weight * *component;
            }
        }
        let scale = direction.dot(&direction).sqrt();
        if scale < 1.0e-12 {
            continue;
        }
        direction /= scale;
        let Some(length) = bisect_rise(
            origin,
            direction.view(),
            start_energy,
            harmonic,
            contact,
            evaluate,
        ) else {
            continue;
        };
        let mut here = origin.to_owned();
        for (value, component) in here.iter_mut().zip(direction.iter()) {
            *value += length * *component;
        }
        if closest_pair(&here) < 0.5 * contact {
            continue;
        }
        let Some(features) =
            curvature_features(here.view(), |point| Some(evaluate(point).1), 12, 1.0e-4)
        else {
            continue;
        };
        let (energy, _) = evaluate(here.view());
        let rise = energy - start_energy;
        if !(rise < harmonic * LEAVE_BARRIER_GROWTH) {
            continue;
        }
        if features.lambda_min < softest {
            softest = features.lambda_min;
            soft_rise = rise;
            soft_point = Some(here);
            soft_mode = Some(features.mode);
        }
    }
    println!("{{\"kind\":\"mix\",\"lambda\":{softest:.4},\"rise\":{soft_rise:.4}}}");
    let _ = std::io::stdout().flush();
    if softest < 0.0
        && let (Some(point), Some(mode)) = (soft_point, soft_mode)
        && push_mode(
            point.view(),
            mode.view(),
            contact,
            soft_rise,
            softest,
            0,
            start_energy,
            ceiling,
            evaluate,
            quench,
            &mut best,
            &mut landed,
            &mut seen,
        )
    {
        return best;
    }
    best
}

/// Covering lines pushed until the energy has risen by one harmonic
/// contact, then a short climb of the softest mode at that point.
fn cover_ridges<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    hops: usize,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let n_dir = (hops.max(1).saturating_mul(8)).min(crate::hypersphere::default_cover_size());
    let mut best = start_energy;
    let mut landed = Vec::new();
    let mut seen = HashSet::new();
    let ceiling = start_energy + harmonic * LEAVE_BARRIER_GROWTH.powi(2);
    let cap = contact / LEAVE_BARRIER_GROWTH.powi(2);
    println!("{{\"kind\":\"ridges\",\"directions\":{n_dir}}}");
    let _ = std::io::stdout().flush();
    for index in 0..n_dir {
        let raw = crate::hypersphere::cover_direction(n_dir.max(1), origin.len(), index);
        let mut direction = Array1::from(raw);
        let norm = direction.dot(&direction).sqrt();
        if norm < 1.0e-12 {
            continue;
        }
        direction /= norm;
        let Some(length) = bisect_rise(
            origin,
            direction.view(),
            start_energy,
            harmonic,
            contact,
            evaluate,
        ) else {
            continue;
        };
        let mut here = origin.to_owned();
        for (value, component) in here.iter_mut().zip(direction.iter()) {
            *value += length * *component;
        }
        let Some(features) =
            curvature_features(here.view(), |point| Some(evaluate(point).1), 12, 1.0e-4)
        else {
            continue;
        };
        let (energy, _) = evaluate(here.view());
        let rise = energy - start_energy;
        println!(
            "{{\"kind\":\"ridge\",\"index\":{index},\"length\":{length:.4},\"rise\":{rise:.4},\"lambda\":{:.4}}}",
            features.lambda_min
        );
        let _ = std::io::stdout().flush();
        if features.lambda_min < 0.0
            && push_mode(
                here.view(),
                features.mode.view(),
                contact,
                rise,
                features.lambda_min,
                index,
                start_energy,
                ceiling,
                evaluate,
                quench,
                &mut best,
                &mut landed,
                &mut seen,
            )
        {
            return best;
        }
        let mut previous = features.mode.clone();
        let mut best_parallel = f64::MAX;
        for _step in 0..12 {
            let Some((lambda, mode)) =
                tracked_mode(here.view(), previous.view(), 8, 1.0e-4, |point| {
                    Some(evaluate(point).1)
                })
            else {
                break;
            };
            previous = mode.clone();
            let (energy, gradient) = evaluate(here.view());
            if !energy.is_finite() {
                break;
            }
            let parallel: f64 = gradient
                .iter()
                .zip(mode.iter())
                .map(|(force, component)| force * component)
                .sum();
            if lambda < 0.0 && parallel.abs() < best_parallel {
                best_parallel = parallel.abs();
                if parallel.abs() < 0.2
                    && push_mode(
                        here.view(),
                        mode.view(),
                        contact,
                        energy - start_energy,
                        lambda,
                        index,
                        start_energy,
                        ceiling,
                        evaluate,
                        quench,
                        &mut best,
                        &mut landed,
                        &mut seen,
                    )
                {
                    return best;
                }
            }
            let parallel_step = if lambda.abs() < 1.0e-8 {
                0.0
            } else {
                (parallel / lambda.abs()).clamp(-cap, cap)
            };
            let mut delta = mode.clone();
            delta *= parallel_step;
            for (value, component) in here.iter_mut().zip(delta.iter()) {
                *value += *component;
            }
            if here.iter().any(|value| !value.is_finite()) || closest_pair(&here) < 0.5 * contact {
                break;
            }
        }
        let quenched = quench(here.view());
        if keep(
            quenched.view(),
            index,
            start_energy,
            ceiling,
            evaluate,
            &mut best,
            &mut landed,
            &mut seen,
        ) {
            return best;
        }
    }
    best
}

fn bisect_rise<E>(
    origin: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    start_energy: f64,
    harmonic: f64,
    contact: f64,
    evaluate: &mut E,
) -> Option<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let mut at = |length: f64| -> Option<f64> {
        let mut point = origin.to_owned();
        for (value, component) in point.iter_mut().zip(direction.iter()) {
            *value += length * *component;
        }
        if closest_pair(&point) < 0.5 * contact {
            return None;
        }
        let (energy, _) = evaluate(point.view());
        energy.is_finite().then_some(energy)
    };
    let mut hi = contact / LEAVE_BARRIER_GROWTH.powi(3);
    let mut hi_rise = 0.0;
    for _ in 0..8 {
        let energy = at(hi)?;
        hi_rise = energy - start_energy;
        if hi_rise >= harmonic {
            break;
        }
        hi *= LEAVE_BARRIER_GROWTH;
        if hi > contact * LEAVE_BARRIER_GROWTH.powi(3) {
            break;
        }
    }
    if hi_rise < harmonic * LEAVE_BARRIER_FLOOR {
        return Some(hi);
    }
    let mut lo = 0.0;
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        let Some(energy) = at(mid) else {
            hi = mid;
            continue;
        };
        if energy - start_energy < harmonic {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Lower bound on the rise, measured from the minimum being climbed.
///
/// A climb that starts on the floor waits out a full rung before a
/// negative curvature is a saddle. A minimum already above that rung can
/// face a shorter saddle, so the bound shrinks by the same ratio.
fn rise_floor(height: f64, start_energy: f64, harmonic: f64) -> f64 {
    if height > start_energy + harmonic * LEAVE_BARRIER_FLOOR {
        harmonic * LEAVE_BARRIER_FLOOR.powi(2)
    } else {
        harmonic * LEAVE_BARRIER_FLOOR
    }
}

/// Climb minima whose energy sits near one harmonic contact above the start.
///
/// The rise of each climb is measured from the minimum being climbed, so a
/// structure already one barrier up can cross the next saddle. The minima
/// nearest that barrier are then climbed on a wider soft subspace.
fn band_search<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let mut best = start_energy;
    let mut queue = vec![(start_energy, origin.to_owned())];
    let mut climbed = HashSet::new();
    let mut climbed_points: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut seen = HashSet::new();
    seen.insert(basin_key(start_energy));
    let mut landed: Vec<(f64, Array1<f64>)> = Vec::new();
    let target = start_energy + harmonic;
    let cap = contact / LEAVE_BARRIER_GROWTH.powi(2);
    let launch_cap = cap;
    let launch_floor = contact / LEAVE_BARRIER_GROWTH.powi(4);
    let mut taken = 0usize;
    while taken < 16 {
        let Some(choice) = queue
            .iter()
            .enumerate()
            .filter(|(_, (energy, _))| !climbed.contains(&basin_key(*energy)))
            .min_by(|(_, a), (_, b)| (a.0 - target).abs().total_cmp(&(b.0 - target).abs()))
            .map(|(index, _)| index)
        else {
            break;
        };
        let (height, point) = queue.swap_remove(choice);
        if !climbed.insert(basin_key(height)) {
            continue;
        }
        taken += 1;
        climbed_points.push((height, point.clone()));
        let ceiling = height + harmonic * LEAVE_BARRIER_GROWTH;
        let floor_rise = rise_floor(height, start_energy, harmonic);
        println!("{{\"kind\":\"band\",\"n\":{taken},\"energy\":{height:.6}}}");
        let _ = std::io::stdout().flush();
        let Some((lambdas, modes, _)) = soft_subspace(
            point.view(),
            |sample| Some(evaluate(sample).1),
            24,
            1.0e-4,
            6,
        ) else {
            continue;
        };
        for (index, mode0) in modes.iter().enumerate() {
            let lambda0 = lambdas.get(index).copied().unwrap_or(1.0).abs().max(1.0);
            let launch = (harmonic / (2.0 * lambda0))
                .sqrt()
                .clamp(launch_floor, launch_cap);
            for sign in [1.0_f64, -1.0] {
                let mut here = point.clone();
                for (value, component) in here.iter_mut().zip(mode0.iter()) {
                    *value += sign * launch * *component;
                }
                let mut previous = mode0.clone();
                if sign < 0.0 {
                    previous *= -1.0;
                }
                let mut best_parallel = f64::MAX;
                for _step in 0..16 {
                    let Some((lambda, mode)) =
                        tracked_mode(here.view(), previous.view(), 8, 1.0e-4, |sample| {
                            Some(evaluate(sample).1)
                        })
                    else {
                        break;
                    };
                    previous = mode.clone();
                    let (energy, gradient) = evaluate(here.view());
                    if !energy.is_finite() {
                        break;
                    }
                    let parallel: f64 = gradient
                        .iter()
                        .zip(mode.iter())
                        .map(|(force, component)| force * component)
                        .sum();
                    let rise = energy - height;
                    if lambda < 0.0 && parallel.abs() < best_parallel {
                        best_parallel = parallel.abs();
                        if parallel.abs() < 0.2
                            && rise > floor_rise
                            && rise < harmonic * LEAVE_BARRIER_GROWTH
                            && push_mode(
                                here.view(),
                                mode.view(),
                                contact,
                                rise,
                                lambda,
                                index,
                                start_energy,
                                ceiling,
                                evaluate,
                                quench,
                                &mut best,
                                &mut landed,
                                &mut seen,
                            )
                        {
                            return best;
                        }
                    }
                    if rise > harmonic * LEAVE_BARRIER_GROWTH {
                        break;
                    }
                    let parallel_step = if lambda.abs() < 1.0e-8 {
                        0.0
                    } else {
                        (parallel / lambda.abs()).clamp(-cap, cap)
                    };
                    let mut delta = mode.clone();
                    delta *= parallel_step;
                    for (value, component) in here.iter_mut().zip(delta.iter()) {
                        *value += *component;
                    }
                    if here.iter().any(|value| !value.is_finite())
                        || closest_pair(&here) < 0.5 * contact
                    {
                        break;
                    }
                }
            }
        }
        for item in landed.drain(..) {
            queue.push(item);
        }
    }
    climbed_points
        .sort_by(|left, right| (left.0 - target).abs().total_cmp(&(right.0 - target).abs()));
    let lip: Vec<(f64, Array1<f64>)> = climbed_points
        .into_iter()
        .filter(|(energy, _)| (*energy - start_energy).abs() > 1.0e-3)
        .take(4)
        .collect();
    if climb_lip(
        lip,
        contact,
        start_energy,
        harmonic,
        evaluate,
        quench,
        &mut best,
    ) {
        return best;
    }
    best
}

/// Wider soft-subspace climb from the minima nearest one harmonic rise.
///
/// Six modes are enough to move inside one funnel. The barrier minimum is
/// climbed again on the wider non-rigid subspace, with the rise measured
/// from that minimum.
fn climb_lip<E, Q>(
    points: Vec<(f64, Array1<f64>)>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    // The softest modes of a barrier minimum point back down its own
    // funnel. The next band is the one that can face the other way.
    let modes_wanted = 40usize;
    let skip_soft = 16usize;
    let cap = contact / LEAVE_BARRIER_GROWTH.powi(2);
    let launch_cap = cap;
    let launch_floor = contact / LEAVE_BARRIER_GROWTH.powi(4);
    let mut landed: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut seen = HashSet::new();
    for (nth, (height, point)) in points.into_iter().enumerate() {
        let ceiling = height + harmonic * LEAVE_BARRIER_GROWTH;
        let floor_rise = rise_floor(height, start_energy, harmonic);
        println!(
            "{{\"kind\":\"lip\",\"n\":{nth},\"energy\":{height:.6},\"modes\":{modes_wanted},\"skip\":{skip_soft}}}"
        );
        let _ = std::io::stdout().flush();
        let Some((lambdas, modes, _)) = soft_subspace(
            point.view(),
            |sample| Some(evaluate(sample).1),
            modes_wanted.saturating_mul(2),
            1.0e-4,
            modes_wanted,
        ) else {
            continue;
        };
        for (index, mode0) in modes.iter().enumerate().skip(skip_soft) {
            let lambda0 = lambdas.get(index).copied().unwrap_or(1.0).abs().max(1.0);
            let launch = (harmonic / (2.0 * lambda0))
                .sqrt()
                .clamp(launch_floor, launch_cap);
            for sign in [1.0_f64, -1.0] {
                let mut here = point.clone();
                for (value, component) in here.iter_mut().zip(mode0.iter()) {
                    *value += sign * launch * *component;
                }
                let mut previous = mode0.clone();
                if sign < 0.0 {
                    previous *= -1.0;
                }
                let mut best_parallel = f64::MAX;
                for _step in 0..16 {
                    let Some((lambda, mode)) =
                        tracked_mode(here.view(), previous.view(), 8, 1.0e-4, |sample| {
                            Some(evaluate(sample).1)
                        })
                    else {
                        break;
                    };
                    previous = mode.clone();
                    let (energy, gradient) = evaluate(here.view());
                    if !energy.is_finite() {
                        break;
                    }
                    let parallel: f64 = gradient
                        .iter()
                        .zip(mode.iter())
                        .map(|(force, component)| force * component)
                        .sum();
                    let rise = energy - height;
                    if lambda < 0.0 && parallel.abs() < best_parallel {
                        best_parallel = parallel.abs();
                        if parallel.abs() < 0.2
                            && rise > floor_rise
                            && rise < harmonic * LEAVE_BARRIER_GROWTH
                            && push_mode(
                                here.view(),
                                mode.view(),
                                contact,
                                rise,
                                lambda,
                                index,
                                start_energy,
                                ceiling,
                                evaluate,
                                quench,
                                best,
                                &mut landed,
                                &mut seen,
                            )
                        {
                            return true;
                        }
                    }
                    if rise > harmonic * LEAVE_BARRIER_GROWTH {
                        break;
                    }
                    let parallel_step = if lambda.abs() < 1.0e-8 {
                        0.0
                    } else {
                        (parallel / lambda.abs()).clamp(-cap, cap)
                    };
                    let mut delta = mode.clone();
                    delta *= parallel_step;
                    for (value, component) in here.iter_mut().zip(delta.iter()) {
                        *value += *component;
                    }
                    if here.iter().any(|value| !value.is_finite())
                        || closest_pair(&here) < 0.5 * contact
                    {
                        break;
                    }
                }
            }
        }
    }
    false
}

/// Overlap-tracked climb of the softest modes, then a plain quench.
///
/// The direction is recomputed from the Hessian at each step and kept by
/// overlap, so it can rotate. A positive curvature is walked uphill. A
/// negative curvature is quenched on both sides.
fn follow_modes<E, Q>(
    origin: ArrayView1<f64>,
    contact: f64,
    start_energy: f64,
    harmonic: f64,
    hops: usize,
    evaluate: &mut E,
    quench: &mut Q,
) -> f64
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let count = hops.max(1).min(16);
    let krylov = (count.saturating_mul(2)).clamp(24, 240);
    let Some((lambdas, modes, _)) = soft_subspace(
        origin,
        |point| Some(evaluate(point).1),
        krylov,
        1.0e-4,
        count,
    ) else {
        return start_energy;
    };
    println!("{{\"kind\":\"follow\",\"modes\":{}}}", modes.len());
    let _ = std::io::stdout().flush();
    let mut best = start_energy;
    let mut landed: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut seen = HashSet::new();
    let ceiling = start_energy + harmonic * LEAVE_BARRIER_GROWTH.powi(2);
    let cap = contact / LEAVE_BARRIER_GROWTH.powi(2);
    let launch_cap = contact / LEAVE_BARRIER_GROWTH.powi(2);
    let launch_floor = contact / LEAVE_BARRIER_GROWTH.powi(4);
    let perp_cap = Activation::default().perp_max_move;
    let perp_rate = Activation::default().perp_rate;
    let rise_cap = harmonic * LEAVE_BARRIER_GROWTH.powi(2);
    let climb_steps = Activation::default().max_steps + Activation::default().lanczos_steps;
    for (index, mode0) in modes.iter().enumerate() {
        let lambda0 = lambdas.get(index).copied().unwrap_or(1.0).abs().max(1.0);
        // A fixed length crushes a stiff mode and barely leaves a soft one.
        let launch = (harmonic / (2.0 * lambda0))
            .sqrt()
            .clamp(launch_floor, launch_cap);
        for sign in [1.0_f64, -1.0] {
            let mut here = origin.to_owned();
            for (value, component) in here.iter_mut().zip(mode0.iter()) {
                *value += sign * launch * *component;
            }
            let mut previous = mode0.clone();
            if sign < 0.0 {
                previous *= -1.0;
            }
            let mut best_parallel = f64::MAX;
            for step in 0..climb_steps {
                let Some((lambda, mode)) =
                    tracked_mode(here.view(), previous.view(), 8, 1.0e-4, |point| {
                        Some(evaluate(point).1)
                    })
                else {
                    break;
                };
                previous = mode.clone();
                let (energy, gradient) = evaluate(here.view());
                if !energy.is_finite() {
                    break;
                }
                let rise = energy - start_energy;
                let parallel: f64 = gradient
                    .iter()
                    .zip(mode.iter())
                    .map(|(force, component)| force * component)
                    .sum();
                println!(
                    "{{\"kind\":\"ef\",\"mode\":{index},\"sign\":{sign},\"step\":{step},\"rise\":{rise:.4},\"lambda\":{lambda:.4},\"parallel\":{parallel:.4}}}"
                );
                let _ = std::io::stdout().flush();
                // Quench from the saddle, where the followed curvature is
                // negative and the parallel force has fallen, not from the
                // first step that merely went unstable.
                if lambda < 0.0 && parallel.abs() < best_parallel {
                    best_parallel = parallel.abs();
                    if parallel.abs() < 0.2
                        && push_mode(
                            here.view(),
                            mode.view(),
                            contact,
                            rise,
                            lambda,
                            index,
                            start_energy,
                            ceiling,
                            evaluate,
                            quench,
                            &mut best,
                            &mut landed,
                            &mut seen,
                        )
                    {
                        return best;
                    }
                }
                if rise > rise_cap {
                    break;
                }
                let parallel_step = if lambda.abs() < 1.0e-8 {
                    0.0
                } else {
                    (parallel / lambda.abs()).clamp(-cap, cap)
                };
                let mut delta = mode.clone();
                delta *= parallel_step;
                let mut perpendicular = gradient.clone();
                for (slot, component) in perpendicular.iter_mut().zip(mode.iter()) {
                    *slot -= parallel * *component;
                }
                let perp_norm = perpendicular
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if perp_norm > 1.0e-12 {
                    let length = (perp_norm * perp_rate).min(perp_cap);
                    let scale = length / perp_norm;
                    for (slot, component) in delta.iter_mut().zip(perpendicular.iter()) {
                        *slot -= scale * *component;
                    }
                }
                for (value, component) in here.iter_mut().zip(delta.iter()) {
                    *value += *component;
                }
                if here.iter().any(|value| !value.is_finite())
                    || closest_pair(&here) < 0.5 * contact
                {
                    break;
                }
            }
            let quenched = quench(here.view());
            if keep(
                quenched.view(),
                index,
                start_energy,
                ceiling,
                evaluate,
                &mut best,
                &mut landed,
                &mut seen,
            ) {
                return best;
            }
        }
    }
    if hops < 4 {
        return best;
    }
    // Several generations from the high side of the window. The first
    // saddle out of the icosahedron stays in its funnel; a later high
    // neighbour can face the other way.
    for generation in 0..4 {
        landed.sort_by(|a, b| b.0.total_cmp(&a.0));
        let neighbours: Vec<(f64, Array1<f64>)> = landed.drain(..).take(4).collect();
        if neighbours.is_empty() {
            break;
        }
        for (nth, (height, neighbour)) in neighbours.into_iter().enumerate() {
            println!(
                "{{\"kind\":\"neighbour\",\"generation\":{generation},\"n\":{nth},\"energy\":{height:.6}}}"
            );
            let _ = std::io::stdout().flush();
            let Some((_, modes, _)) = soft_subspace(
                neighbour.view(),
                |point| Some(evaluate(point).1),
                24,
                1.0e-4,
                4,
            ) else {
                continue;
            };
            for (index, mode0) in modes.iter().enumerate() {
                for sign in [1.0_f64, -1.0] {
                    let mut here = neighbour.clone();
                    for (value, component) in here.iter_mut().zip(mode0.iter()) {
                        *value += sign * launch_cap * *component;
                    }
                    let mut previous = mode0.clone();
                    if sign < 0.0 {
                        previous *= -1.0;
                    }
                    let mut best_parallel = f64::MAX;
                    for _step in 0..climb_steps {
                        let Some((lambda, mode)) =
                            tracked_mode(here.view(), previous.view(), 8, 1.0e-4, |point| {
                                Some(evaluate(point).1)
                            })
                        else {
                            break;
                        };
                        previous = mode.clone();
                        let (energy, gradient) = evaluate(here.view());
                        if !energy.is_finite() {
                            break;
                        }
                        let parallel: f64 = gradient
                            .iter()
                            .zip(mode.iter())
                            .map(|(force, component)| force * component)
                            .sum();
                        if lambda < 0.0 && parallel.abs() < best_parallel {
                            best_parallel = parallel.abs();
                            if parallel.abs() < 0.2
                                && push_mode(
                                    here.view(),
                                    mode.view(),
                                    contact,
                                    energy - start_energy,
                                    lambda,
                                    index,
                                    start_energy,
                                    ceiling,
                                    evaluate,
                                    quench,
                                    &mut best,
                                    &mut landed,
                                    &mut seen,
                                )
                            {
                                return best;
                            }
                        }
                        if energy - start_energy > rise_cap {
                            break;
                        }
                        let parallel_step = if lambda.abs() < 1.0e-8 {
                            0.0
                        } else {
                            (parallel / lambda.abs()).clamp(-cap, cap)
                        };
                        let mut delta = mode.clone();
                        delta *= parallel_step;
                        for (value, component) in here.iter_mut().zip(delta.iter()) {
                            *value += *component;
                        }
                        if here.iter().any(|value| !value.is_finite())
                            || closest_pair(&here) < 0.5 * contact
                        {
                            break;
                        }
                    }
                    let quenched = quench(here.view());
                    if keep(
                        quenched.view(),
                        index,
                        start_energy,
                        ceiling,
                        evaluate,
                        &mut best,
                        &mut landed,
                        &mut seen,
                    ) {
                        return best;
                    }
                }
            }
        }
    }
    best
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

fn push_mode<E, Q>(
    point: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    contact: f64,
    rise: f64,
    lambda: f64,
    hop: usize,
    start_energy: f64,
    ceiling: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
    landed: &mut Vec<(f64, Array1<f64>)>,
    seen: &mut HashSet<usize>,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    // Harmonic distance from this curvature down a drop the size of the
    // climb. A shorter push is still on the side the climb came from.
    let drop = (2.0 * rise.max(0.0) / lambda.abs().max(1.0e-6)).sqrt();
    let span = drop.clamp(
        contact / LEAVE_BARRIER_GROWTH.powi(2),
        contact * LEAVE_BARRIER_GROWTH.powi(2),
    );
    for sign in [1.0_f64, -1.0] {
        for length in [span / LEAVE_BARRIER_GROWTH, span] {
            let mut far = point.to_owned();
            for (value, component) in far.iter_mut().zip(mode.iter()) {
                *value += sign * length * *component;
            }
            if far.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let quenched = quench(far.view());
            if keep(
                quenched.view(),
                hop,
                start_energy,
                ceiling,
                evaluate,
                best,
                landed,
                seen,
            ) {
                return true;
            }
        }
    }
    false
}

/// Records a plain quench. A minimum above the start and under `ceiling`
/// is kept so the next generation can climb from it.
fn keep<E>(
    coords: ArrayView1<f64>,
    hop: usize,
    start_energy: f64,
    ceiling: f64,
    evaluate: &mut E,
    best: &mut f64,
    landed: &mut Vec<(f64, Array1<f64>)>,
    seen: &mut HashSet<usize>,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let (energy, _) = evaluate(coords);
    if !energy.is_finite() {
        return false;
    }
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
    );
    let _ = std::io::stdout().flush();
    if energy < *best {
        *best = energy;
    }
    if *best < start_energy - 1.0e-4 {
        return true;
    }
    let key = basin_key(energy);
    if energy > start_energy + 1.0e-3 && energy < ceiling && seen.insert(key) {
        landed.push((energy, coords.to_owned()));
    }
    false
}

/// Covering launch, softened microcanonical escape, and a plain quench.
///
/// The direction is one vector of the covering. A short modified-dimer
/// softening bends it toward the low curvature without collapsing every
/// launch onto the softest mode. The trajectory stops at the next potential
/// minimum, and that minimum is quenched on the plain energy. Goedecker's
/// history raises the kick after a return and lowers it after a discovery.
/// The kick stays at least the harmonic cost of one contact.
fn softened_chain<E, Q>(
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
    Q: FnMut(ArrayView1<f64>) -> Array1<f64> + Send,
{
    if hops == 0 {
        return start_energy;
    }
    // A kick as large as the contact harmonic steps over the low
    // barriers. The gentle rung is the one that crosses into a
    // neighbour of similar energy. The cohesive energy is only the cap.
    let gentle = harmonic * LEAVE_BARRIER_FLOOR;
    let mut feedback = EscapeFeedback::new(gentle, gentle);
    feedback.escape_floor = gentle;
    feedback.escape_ceiling = start_energy.abs();
    let n_atoms = (origin.len() / 3).max(1) as f64;
    let soften_steps = Activation::default().lanczos_steps * LEAVE_BARRIER_GROWTH.powi(2) as usize;
    let well_rms = contact / n_atoms.sqrt();
    let escape = MdEscapeConfig {
        geometry: MdEscapeGeometry::RigidQuotient,
        minimum_rise: gentle,
        softening: Some(rgsaddle::VelocitySofteningConfig {
            steps: soften_steps,
            displacement: contact / n_atoms,
            mixing: rgsaddle::VelocitySofteningConfig::default().mixing,
        }),
        potential_minima: 2,
        maximum_steps: MdEscapeConfig::default().maximum_steps,
        max_rms: f64::INFINITY,
        min_well_rms: well_rms,
        ..MdEscapeConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut walker = origin.to_owned();
    let mut walker_energy = start_energy;
    let mut current = basin_key(walker_energy);
    feedback.register_initial(current);
    let mut best = start_energy;
    let n_cover = crate::hypersphere::default_cover_size();
    println!(
        "{{\"kind\":\"softened\",\"hops\":{hops},\"escape\":{:.4},\"floor\":{:.4},\"ceiling\":{:.4},\"soften\":{soften_steps},\"well_rms\":{well_rms:.4}}}",
        feedback.escape(),
        feedback.escape_floor,
        feedback.escape_ceiling
    );
    let _ = std::io::stdout().flush();
    for hop in 0..hops {
        let mut direction = Array1::from(crate::hypersphere::cover_direction(
            n_cover,
            walker.len(),
            hop,
        ));
        let norm = direction.dot(&direction).sqrt();
        if norm > 1.0e-12 {
            direction /= norm;
        }
        let kinetic = feedback.escape();
        // A hard kick reaches the contact root-mean-square while it is
        // still a vibration. The distance required to count a new well
        // grows with the square root of the kick, and stops at the
        // cluster's own radius.
        let mut hop_escape = escape;
        hop_escape.min_well_rms = (well_rms * (kinetic / gentle).sqrt()).min(spread(walker.view()));
        let mut surface = |point: ArrayView1<f64>| {
            let (energy, gradient) = evaluate(point);
            if energy.is_finite() && gradient.iter().all(|value| value.is_finite()) {
                Some((energy, gradient))
            } else {
                None
            }
        };
        let escaped = nve_escape_seeded(
            walker.view(),
            kinetic,
            Some(direction.view()),
            &hop_escape,
            &mut surface,
            &mut rng,
        );
        let Ok(report) = escaped else {
            println!("{{\"kind\":\"escape\",\"hop\":{hop},\"ok\":false}}");
            let _ = std::io::stdout().flush();
            continue;
        };
        let landed = if report.potential_minima > 0 {
            report.position.clone()
        } else {
            report.far_position.clone()
        };
        // The well quench is the move. The farthest point is quenched
        // as well, and only the record of the lowest plain energy uses it.
        let mut energy = f64::NAN;
        let mut quenched = landed.clone();
        for (index, trial) in [landed, report.far_position].into_iter().enumerate() {
            let relaxed = quench(trial.view());
            let (trial_energy, _) = evaluate(relaxed.view());
            if !trial_energy.is_finite() {
                continue;
            }
            println!(
                "{{\"kind\":\"exit_candidate\",\"energy\":{trial_energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
            );
            if trial_energy < best {
                best = trial_energy;
            }
            if index == 0 {
                energy = trial_energy;
                quenched = relaxed;
            }
        }
        let _ = std::io::stdout().flush();
        if !energy.is_finite() {
            continue;
        }
        if best < start_energy - 1.0e-4 {
            println!(
                "{{\"kind\":\"hop\",\"hop\":{hop},\"energy\":{energy:.6},\"walker\":{walker_energy:.6},\"md_steps\":{},\"minima\":{},\"far_rms\":{:.4}}}",
                report.steps, report.potential_minima, report.far_rms
            );
            let _ = std::io::stdout().flush();
            return best;
        }
        let key = basin_key(energy);
        let visit = feedback.observe(Some(current), key);
        let delta = energy - walker_energy;
        let accepted = if visit == Visit::Same {
            false
        } else {
            feedback.accept(delta)
        };
        println!(
            "{{\"kind\":\"hop\",\"hop\":{hop},\"energy\":{energy:.6},\"walker\":{walker_energy:.6},\"delta\":{delta:.4},\"escape\":{:.4},\"threshold\":{:.4},\"visit\":\"{}\",\"accepted\":{accepted},\"md_steps\":{},\"minima\":{},\"far_rms\":{:.4}}}",
            feedback.escape(),
            feedback.threshold(),
            visit_name(visit),
            report.steps,
            report.potential_minima,
            report.far_rms
        );
        let _ = std::io::stdout().flush();
        if accepted {
            walker = quenched;
            walker_energy = energy;
            current = key;
        }
    }
    best
}

fn hop_chain<E, Q>(
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
    Q: FnMut(ArrayView1<f64>) -> Array1<f64> + Send,
{
    if hops == 0 {
        return start_energy;
    }
    let mut feedback = EscapeFeedback::new(harmonic, harmonic);
    // A kick softer than the contact harmonic cost falls back into the
    // same well. Discoveries must not shrink the escape through that floor.
    // The ceiling is the cohesive energy. A kick of that size can disorder
    // the cluster; a quarter of it stays inside the icosahedral catchment.
    feedback.escape_floor = harmonic;
    feedback.escape_ceiling = start_energy.abs();
    let mut walker = origin.to_owned();
    let mut walker_energy = start_energy;
    let mut current = basin_key(walker_energy);
    feedback.register_initial(current);
    let mut best = start_energy;
    let mut escape = MdEscapeConfig {
        geometry: MdEscapeGeometry::RigidQuotient,
        minimum_rise: harmonic * LEAVE_BARRIER_FLOOR,
        // The climb is the minimum-mode pass. The escape carries the kick
        // on one soft mode and stops when the cluster has moved by its
        // own radius.
        softening: None,
        potential_minima: usize::MAX,
        maximum_steps: MdEscapeConfig::default().maximum_steps
            * LEAVE_BARRIER_GROWTH.powi(2) as usize,
        max_rms: spread(origin),
        min_well_rms: 0.0,
        ..MdEscapeConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let climb = Activation::default();
    let n_cover = crate::hypersphere::default_cover_size();
    println!(
        "{{\"kind\":\"chain\",\"hops\":{hops},\"escape\":{harmonic:.4},\"floor\":{:.4},\"ceiling\":{:.4},\"max_rms\":{:.4}}}",
        feedback.escape_floor, feedback.escape_ceiling, escape.max_rms
    );
    let _ = std::io::stdout().flush();
    for hop in 0..hops {
        let index = hop.wrapping_add(seed as usize);
        let climbed = cover_climb_quench(
            walker.view(),
            contact,
            index,
            |point| {
                let (_, gradient) = evaluate(point);
                Some(gradient)
            },
            |point| quench(point),
            &climb,
        );
        if probe(climbed.view(), hop, start_energy, evaluate, &mut best) {
            return best;
        }
        let mut direction = Array1::from(crate::hypersphere::cover_direction(
            n_cover,
            walker.len(),
            index,
        ));
        // The cover index selects one soft mode. A full-space kick rattles
        // stiff bonds and the root-mean-square displacement stays tiny.
        if let Some((_, modes, _)) = soft_subspace(
            walker.view(),
            |point| Some(evaluate(point).1),
            24,
            1.0e-4,
            Activation::default().lanczos_steps.max(4),
        ) {
            let mode = &modes[index % modes.len()];
            let mut aimed = mode.clone();
            if index % 2 == 0 {
                for component in aimed.iter_mut() {
                    *component = -*component;
                }
            }
            direction = aimed;
        }
        let kinetic = feedback.escape();
        // A counted minimum has to fall by the ladder floor times the
        // contact harmonic cost. Scaling that rise with the kick hides
        // the barrier the escape is trying to cross.
        escape.minimum_rise = harmonic * LEAVE_BARRIER_FLOOR;
        let mut surface = |point: ArrayView1<f64>| {
            let (energy, gradient) = evaluate(point);
            if energy.is_finite() && gradient.iter().all(|value| value.is_finite()) {
                Some((energy, gradient))
            } else {
                None
            }
        };
        let escaped = nve_escape_seeded(
            walker.view(),
            kinetic,
            Some(direction.view()),
            &escape,
            &mut surface,
            &mut rng,
        );
        let Ok(report) = escaped else {
            println!("{{\"kind\":\"escape\",\"hop\":{hop},\"ok\":false}}");
            let _ = std::io::stdout().flush();
            continue;
        };
        let n_atoms = walker.len() / 3;
        let mut shift = 0.0;
        if n_atoms > 0 {
            for (there, here) in report.position.iter().zip(walker.iter()) {
                let delta = there - here;
                shift += delta * delta;
            }
            shift = (shift / n_atoms as f64).sqrt();
        }
        println!(
            "{{\"kind\":\"escape\",\"hop\":{hop},\"steps\":{},\"minima\":{},\"md_energy\":{:.4},\"kinetic\":{:.4},\"rms\":{shift:.4},\"far_rms\":{:.4}}}",
            report.steps, report.potential_minima, report.energy, report.kinetic, report.far_rms
        );
        // Quench the well the escape stopped in. A negative mode there
        // is pushed both ways before that quench is accepted.
        if report.potential_minima == 0
            && report.far_rms > contact / (n_atoms.max(1) as f64).sqrt()
            && push_unstable(
                report.position.view(),
                contact,
                hop,
                start_energy,
                evaluate,
                quench,
                &mut best,
            )
        {
            return best;
        }
        let quenched = quench(report.far_position.view());
        if accept_escape(
            quenched.view(),
            hop,
            start_energy,
            evaluate,
            &mut best,
            &mut feedback,
            &mut walker,
            &mut walker_energy,
            &mut current,
        ) {
            return best;
        }
    }
    best
}

/// Both sides of a negative mode at `point`, on the plain energy.
///
/// Returns whether a quench landed strictly below the start.
fn push_unstable<E, Q>(
    point: ArrayView1<f64>,
    contact: f64,
    hop: usize,
    start_energy: f64,
    evaluate: &mut E,
    quench: &mut Q,
    best: &mut f64,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let Some(features) = curvature_features(point, |there| Some(evaluate(there).1), 24, 1.0e-4)
    else {
        return false;
    };
    println!(
        "{{\"kind\":\"curvature\",\"hop\":{hop},\"lambda\":{:.4}}}",
        features.lambda_min
    );
    let _ = std::io::stdout().flush();
    if !(features.lambda_min < 0.0) {
        return false;
    }
    for sign in [1.0_f64, -1.0] {
        for length in [contact, contact * LEAVE_BARRIER_GROWTH] {
            let mut far = point.to_owned();
            for (value, component) in far.iter_mut().zip(features.mode.iter()) {
                *value += sign * length * *component;
            }
            if far.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let quenched = quench(far.view());
            if probe(quenched.view(), hop, start_energy, evaluate, best) {
                return true;
            }
        }
    }
    false
}

/// Plain quench of a climb. The climb does not move the chain: a high
/// landing must not shrink the kick or loosen the acceptance threshold.
fn probe<E>(
    coords: ArrayView1<f64>,
    hop: usize,
    start_energy: f64,
    evaluate: &mut E,
    best: &mut f64,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let (energy, _) = evaluate(coords);
    if !energy.is_finite() {
        return false;
    }
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
    );
    let _ = std::io::stdout().flush();
    if energy < *best {
        *best = energy;
    }
    *best < start_energy - 1.0e-4
}

/// Quench of one escape, then Goedecker feedback. Returns whether the best
/// plain energy is strictly below the start.
fn accept_escape<E>(
    coords: ArrayView1<f64>,
    hop: usize,
    start_energy: f64,
    evaluate: &mut E,
    best: &mut f64,
    feedback: &mut EscapeFeedback,
    walker: &mut Array1<f64>,
    walker_energy: &mut f64,
    current: &mut usize,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let (energy, _) = evaluate(coords);
    if !energy.is_finite() {
        return false;
    }
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{energy:.6},\"hop\":{hop},\"role\":\"quench\"}}"
    );
    if energy < *best {
        *best = energy;
    }
    let key = basin_key(energy);
    let visit = feedback.observe(Some(*current), key);
    let delta = energy - *walker_energy;
    // Returning to the current basin is not a move. Touching the
    // threshold on it would drive the threshold to zero and then refuse
    // every real step out.
    let accepted = if visit == Visit::Same {
        false
    } else {
        feedback.accept(delta)
    };
    println!(
        "{{\"kind\":\"hop\",\"hop\":{hop},\"role\":\"escape\",\"energy\":{energy:.6},\"walker\":{:.6},\"delta\":{delta:.4},\"escape\":{:.4},\"visit\":\"{}\",\"accepted\":{accepted}}}",
        *walker_energy,
        feedback.escape(),
        visit_name(visit)
    );
    let _ = std::io::stdout().flush();
    if accepted {
        *walker = coords.to_owned();
        *walker_energy = energy;
        *current = key;
    }
    *best < start_energy - 1.0e-4
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
        let best = search(quenched.view(), contact, 2000, 1, lj, quench);
        assert!(best.is_finite(), "plain quench was not finite");
        assert!(
            best < -396.282249,
            "plain quench {best:.6} did not fall below the icosahedron {start:.6}"
        );
    }
}
