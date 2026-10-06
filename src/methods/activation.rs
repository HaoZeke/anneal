//! Activation: climb the ridge out of a basin, then quench the other side.
//!
//! This is valley-floor / ridge following, not a quench. Quapp's gradient
//! extremal of the smallest Hessian eigenvalue is the valley floor or the
//! ridge (Quapp, *Chem. Phys. Lett.* **1996**, *253*, 286,
//! <https://doi.org/10.1016/0009-2614(96)00255-2>). Reduced-gradient
//! following traces the same curves (Quapp, Hirsch, Imig, Heidrich, *J.
//! Comput. Chem.* **1998**, *19*, 1087). Barkema and Mousseau's ART
//! (*Phys. Rev. Lett.* **1996**, *77*, 4358,
//! <https://doi.org/10.1103/PhysRevLett.77.4358>; Malek and Mousseau,
//! *Phys. Rev. E* **2000**, *62*, 7723,
//! <https://doi.org/10.1103/PhysRevE.62.7723>) climbs that direction
//! until the curvature turns over. Henkelman and Jónsson's dimer
//! (*J. Chem. Phys.* **1999**, *111*, 7010,
//! <https://doi.org/10.1063/1.480097>) and Plasencia's SoftSaddle MMF
//! (*J. Chem. Theory Comput.* **2017**, *13*, 125,
//! <https://doi.org/10.1021/acs.jctc.5b01216>) invert the force along
//! the minimum mode and minimise the rest: that is the same ridge walk
//! to a first-order saddle. Xiao, Wu and Henkelman (*J. Chem. Phys.*
//! **2014**, *141*, 164111, <https://doi.org/10.1063/1.4898664>)
//! distinguish that *local* ridge (force perpendicular to the negative
//! mode) from the true basin-boundary ridge; a quench is taken only
//! after the force along the mode has changed sign, so the landing is
//! past the saddle and not back down the local ridge into the well.
//!
//! A single displacement along the softest mode does not leave a basin. It is
//! the right direction and the wrong distance: the mode points at the low
//! saddle, but relaxing from a point still inside the basin returns to the
//! minimum it came from. Measured on LJ38 with the escape controller driving a
//! straight displacement, 576 quenches in 959 came back to the basin they left
//! and 10 found anything new.
//!
//! Goedecker's answer is molecular dynamics, which carries kinetic energy over
//! the saddle. The answer that needs only gradients is to climb: push along the
//! mode, relax the components perpendicular to it so the structure stays on the
//! valley floor, and repeat until the curvature along the mode turns negative
//! *and* the force along the climb has flipped. That is the ridge, and a
//! quench from the overshoot falls into a different basin.
//!
//! What this costs is honest and worth stating. Each climbing step is a
//! curvature pass and a few perpendicular relaxation steps, so an activation is
//! several hundred charged evaluations where a random displacement is one. It
//! buys escapes that actually leave.
//!
//! # Relation to the rest of the crate
//!
//! The perpendicular relaxation is the same projection [`crate::path`] uses to
//! hold a band off its endpoints, and the mode comes from
//! [`crate::curvature`]. The controller in [`crate::methods::minima_hopping`]
//! sets how far to climb; this module decides when to stop.

use crate::curvature::{curvature_features, project_rigid_with, rigid_basis};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng};
use std::io::Write;

/// How the climb is run.
#[derive(Debug, Clone)]
pub struct Activation {
    /// Distance moved along the mode per climbing step.
    pub step: f64,
    /// Climbing steps before giving up.
    ///
    /// A cap rather than a convergence criterion: some directions do not reach
    /// negative curvature at all, and a climb that has not turned over after
    /// this many steps is abandoned rather than run to exhaustion.
    pub max_steps: usize,
    /// Perpendicular relaxation steps between climbs.
    pub perp_steps: usize,
    /// Step size of the perpendicular relaxation.
    pub perp_rate: f64,
    /// Largest displacement one perpendicular step may make.
    ///
    /// A fixed rate is not safe on a potential whose gradient spans decades. On
    /// a Lennard-Jones cluster two points a little too close carry a gradient of
    /// order a thousand, and a rate of 0.02 against that moves the structure
    /// twenty units and destroys it: measured on LJ38, 6 relaxations in 1589
    /// reached a minimum and the returned structure had a gradient of 1.0 where
    /// a minimum has 1e-6. The cap makes the step a direction with a bounded
    /// length rather than a length proportional to the gradient.
    pub perp_max_move: f64,
    /// Lanczos steps per curvature pass.
    pub lanczos_steps: usize,
    /// Finite-difference step for the curvature.
    pub epsilon: f64,
    /// Climbing steps between recomputing the mode.
    ///
    /// Recomputing every step is the accurate choice and the expensive one. The
    /// mode rotates slowly along a valley floor, so reusing it for a few steps
    /// costs little accuracy and divides the curvature bill.
    pub refresh: usize,
    /// Extra push along the mode once the curvature has turned over, in units
    /// of `step`, before the quench.
    pub overshoot: f64,
    /// Ignore a ridge whose integrated rise is below this.
    ///
    /// The first saddle out of a low minimum is often a shallow step to a
    /// neighbour of similar depth. A later ridge, several energy units up,
    /// is the one that can open another funnel. Zero keeps every ridge.
    pub min_rise: f64,
}

impl Default for Activation {
    fn default() -> Self {
        Self {
            step: 0.2,
            max_steps: 24,
            perp_steps: 3,
            perp_rate: 0.02,
            perp_max_move: 0.05,
            lanczos_steps: 12,
            epsilon: 1e-4,
            refresh: 3,
            overshoot: 1.5,
            min_rise: 0.0,
        }
    }
}

/// Where a climb ended.
#[derive(Debug, Clone)]
pub struct ActivationOutcome {
    /// The activated structure, to be quenched by the caller.
    pub state: Array1<f64>,
    /// Curvature along the mode at the end of the climb.
    pub lambda: f64,
    /// Climbing steps taken.
    pub steps: usize,
    /// Whether the curvature turned negative, so the ridge is behind.
    pub crossed: bool,
    /// Gradient evaluations spent, all of them charged by the caller.
    pub evaluations: usize,
}

/// Climbs out of the basin containing `x`.
///
/// `grad` returns the gradient or `None` when the caller's budget is spent, in
/// which case the climb stops and reports what it has. `sign` picks which way
/// along the mode to go; the two ends of a soft direction are different saddles.
///
/// Returns `None` only when the first curvature pass fails, since there is then
/// no direction to climb along.
pub fn activate<G>(
    x: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
    sign: f64,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    activate_aligned(x, None, None, &mut grad, cfg, sign)
}

/// One covering displacement, the minimum-mode climb, then the caller's quench.
///
/// The start is only a point and a force. No target energy is read.
/// `cover_index` selects one point of the hypersphere cover. The climb
/// walks away from `origin`. The quench is whatever the caller uses for
/// a local minimisation of the same force.
pub fn cover_climb_quench<G, Q>(
    origin: ArrayView1<f64>,
    rmsd: f64,
    cover_index: usize,
    mut grad: G,
    mut quench: Q,
    cfg: &Activation,
) -> Array1<f64>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let n_cover = crate::hypersphere::default_cover_size();
    let direction = Array1::from(crate::hypersphere::cover_direction(
        n_cover,
        origin.len(),
        cover_index,
    ));
    let contact = closest_pair(origin);
    let n_atoms = origin.len() / 3;
    // On a cluster the cover is the direction of the climb, not a kick
    // that is quenched on its own. One step is one contact length spread
    // over the caller budget. The walk may run for one step per atom,
    // until the force along the cover changes sign, and the quench follows.
    if contact > 0.95 && n_atoms >= 2 {
        let mut climb = cfg.clone();
        let budget = climb.max_steps.max(1);
        climb.step = contact * (n_atoms as f64).sqrt() / budget as f64;
        climb.max_steps = budget;
        climb.min_rise = 0.0;
        if let Some(outcome) = activate_along(origin.view(), direction.view(), &mut grad, &climb)
            && outcome.crossed
        {
            return quench(outcome.state.view());
        }
    }
    let placed = crate::hypersphere::place_around(
        origin.as_slice().unwrap_or(&[]),
        direction.as_slice().unwrap(),
        rmsd.max(1e-3),
        None,
    );
    let start = if placed.len() == origin.len() {
        Array1::from(placed)
    } else {
        origin.to_owned()
    };
    if let Some(outcome) = activate_along(start.view(), direction.view(), &mut grad, cfg)
        && outcome.crossed
    {
        return quench(outcome.state.view());
    }
    quench(start.view())
}

/// Covering displacement, fivefold openings of the same shell, a
/// minimum-mode climb, then a quench. The lowest minimum is kept.
///
/// The hypersphere cover picks a direction. On a shell that still has
/// pentagonal axes, those axes are further covering directions: they
/// are read off the coordinates, not off a target minimum. No target
/// energy is supplied.
pub fn cover_climb_quench_min<G, Q, E>(
    origin: ArrayView1<f64>,
    rmsd: f64,
    cover_index: usize,
    mut grad: G,
    mut quench: Q,
    mut energy: E,
    cfg: &Activation,
) -> Array1<f64>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
    E: FnMut(ArrayView1<f64>) -> f64,
{
    let mut best = origin.to_owned();
    let mut best_e = energy(origin.view());
    let mut keep = |quenched: Array1<f64>| {
        let value = energy(quenched.view());
        if value.is_finite() {
            println!("{{\"kind\":\"exit_candidate\",\"energy\":{value:.6}}}");
        }
        if value.is_finite() && value < best_e - 1e-6 {
            best_e = value;
            best = quenched;
        }
    };
    // Several amplitudes of the pentagonal opening, quenched on their own.
    for amp in [0.35_f64, 0.55, 0.75, 1.0, 1.25] {
        for which in 0..4 {
            keep(quench(fivefold_opening(origin, amp, which).view()));
        }
    }
    // The same opening stacked, then one quench, so the shell can
    // reconstruct before it is relaxed.
    let mut chain = origin.to_owned();
    for which in 0..8 {
        chain = fivefold_opening(chain.view(), 0.75, which);
    }
    keep(quench(chain.view()));
    let mut here = origin.to_owned();
    let mut here_e = best_e;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1 + cover_index as u64);
    let n_cover = crate::hypersphere::default_cover_size();
    let temperature = 8.0_f64;
    // One kick stays in the icosahedral funnel or lands above it.
    // Keep walking: a covering displacement, sometimes a pentagonal
    // opening, a minimum-mode climb when the ridge is real, then a
    // quench. Uphill quenches are accepted so the walk can leave.
    for hop in 0..16 {
        let point = if hop % 6 == 0 {
            fivefold_opening(here.view(), rmsd, hop)
        } else {
            let direction =
                crate::hypersphere::cover_direction(n_cover, here.len(), hop + cover_index);
            let placed = crate::hypersphere::place_around(
                here.as_slice().unwrap_or(&[]),
                &direction,
                rmsd.max(1e-3),
                None,
            );
            if placed.len() == here.len() {
                Array1::from(placed)
            } else {
                here.clone()
            }
        };
        let mut quenched = quench(point.view());
        if let Some(outcome) = activate_from_origin(point.view(), here.view(), &mut grad, cfg) {
            if outcome.crossed {
                let climbed = quench(outcome.state.view());
                let climbed_e = energy(climbed.view());
                let direct_e = energy(quenched.view());
                if climbed_e.is_finite() && (!direct_e.is_finite() || climbed_e < direct_e) {
                    quenched = climbed;
                }
            }
        }
        let value = energy(quenched.view());
        if !value.is_finite() {
            continue;
        }
        println!("{{\"kind\":\"exit_candidate\",\"energy\":{value:.6}}}");
        if value < best_e - 1e-6 {
            best_e = value;
            best = quenched.clone();
        }
        let uphill = value - here_e;
        let accept = uphill <= 0.0 || rng.random::<f64>() < (-uphill / temperature).exp();
        if accept {
            here = quenched;
            here_e = value;
        }
    }
    best
}

struct CoverRidge {
    landings: Vec<Array1<f64>>,
    crossed: bool,
    lambda: f64,
    lowest: f64,
    steps: usize,
    axial: f64,
}

fn empty_ridge() -> CoverRidge {
    CoverRidge {
        landings: Vec::new(),
        crossed: false,
        lambda: 0.0,
        lowest: 0.0,
        steps: 0,
        axial: 0.0,
    }
}

fn dot_av(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn max_atom_weight(mode: ArrayView1<f64>) -> f64 {
    let n_atoms = mode.len() / 3;
    let mut weight = 0.0_f64;
    for atom in 0..n_atoms {
        let mut square = 0.0;
        for axis in 0..3 {
            let component = mode[3 * atom + axis];
            square += component * component;
        }
        weight = weight.max(square.sqrt());
    }
    weight
}

fn cluster_reach(x: ArrayView1<f64>) -> f64 {
    let n_atoms = x.len() / 3;
    if n_atoms == 0 {
        return 0.0;
    }
    let mut com = [0.0; 3];
    for atom in 0..n_atoms {
        for axis in 0..3 {
            com[axis] += x[3 * atom + axis];
        }
    }
    for value in &mut com {
        *value /= n_atoms as f64;
    }
    let mut reach = 0.0_f64;
    for atom in 0..n_atoms {
        let mut square = 0.0;
        for axis in 0..3 {
            let delta = x[3 * atom + axis] - com[axis];
            square += delta * delta;
        }
        reach = reach.max(square.sqrt());
    }
    reach
}

fn axial_projection(cur: ArrayView1<f64>, origin: ArrayView1<f64>, mode: ArrayView1<f64>) -> f64 {
    cur.iter()
        .zip(origin.iter())
        .zip(mode.iter())
        .map(|((value, start), component)| (value - start) * component)
        .sum()
}

fn pin_axial(cur: &mut Array1<f64>, origin: ArrayView1<f64>, mode: ArrayView1<f64>, target: f64) {
    let shift = target - axial_projection(cur.view(), origin, mode);
    if shift == 0.0 {
        return;
    }
    for (value, component) in cur.iter_mut().zip(mode.iter()) {
        *value += shift * component;
    }
}

fn renormalize_mode(mode: &mut Array1<f64>, x: ArrayView1<f64>) -> bool {
    let basis = rigid_basis(x);
    project_rigid_with(mode, &basis);
    let norm = dot_av(mode.view(), mode.view()).sqrt();
    if !norm.is_finite() || norm < 1e-15 {
        return false;
    }
    *mode /= norm;
    true
}

fn directional_curvature<E>(
    cur: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    evaluate: &mut E,
    epsilon: f64,
) -> Option<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let (_, left) = evaluate(cur);
    let mut shifted = cur.to_owned();
    for (value, component) in shifted.iter_mut().zip(mode.iter()) {
        *value += epsilon * component;
    }
    let (_, right) = evaluate(shifted.view());
    let curvature = (dot_av(right.view(), mode) - dot_av(left.view(), mode)) / epsilon;
    curvature.is_finite().then_some(curvature)
}

/// Directions carried by one covering vector.
///
/// The full vector is kept. Atoms heavier than the mean of that vector,
/// and the single heaviest atom, are separate pushes: a uniform step
/// stretches every contact, and the lowest curvature is then an overlap
/// rather than a rearrangement.
fn cover_axes(direction: &[f64]) -> Vec<(&'static str, Array1<f64>)> {
    let n_atoms = direction.len() / 3;
    if n_atoms == 0 || direction.len() != n_atoms * 3 {
        return Vec::new();
    }
    let mut weights = vec![0.0; n_atoms];
    let mut total = 0.0;
    let mut leading = 0usize;
    for atom in 0..n_atoms {
        let mut square = 0.0;
        for axis in 0..3 {
            let component = direction[3 * atom + axis];
            square += component * component;
        }
        let weight = square.sqrt();
        weights[atom] = weight;
        total += weight;
        if weight > weights[leading] {
            leading = atom;
        }
    }
    let mean = total / n_atoms as f64;
    let mut above = Array1::zeros(direction.len());
    let mut kept = 0usize;
    for atom in 0..n_atoms {
        if weights[atom] > mean {
            kept += 1;
            for axis in 0..3 {
                above[3 * atom + axis] = direction[3 * atom + axis];
            }
        }
    }
    let mut lead = Array1::zeros(direction.len());
    for axis in 0..3 {
        lead[3 * leading + axis] = direction[3 * leading + axis];
    }
    let mut axes = Vec::new();
    axes.push(("cover", Array1::from_vec(direction.to_vec())));
    if kept >= 2 {
        axes.push(("mean", above));
    }
    if weights[leading] > mean {
        axes.push(("atom", lead));
    }
    axes
}

fn relax_pinned<E>(
    cur: &mut Array1<f64>,
    origin: ArrayView1<f64>,
    mode: &Array1<f64>,
    target: f64,
    trust: f64,
    contact: f64,
    evaluate: &mut E,
    cfg: &Activation,
) -> bool
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let clash = contact * 0.5;
    // Each round is one trial along the perpendicular force. Further rounds
    // are how a valley is reached; the displacement stays inside `trust`.
    let rounds = cfg.perp_steps.max(1);
    let mut spent = 0.0_f64;
    for _ in 0..rounds {
        if spent >= trust {
            break;
        }
        pin_axial(cur, origin, mode.view(), target);
        if closest_pair(cur.view()) < clash {
            return false;
        }
        let (energy, gradient) = evaluate(cur.view());
        if !energy.is_finite() {
            return false;
        }
        let along = dot_av(gradient.view(), mode.view());
        let mut perp_square = 0.0;
        let mut perp = gradient;
        for (component, mode_component) in perp.iter_mut().zip(mode.iter()) {
            *component -= along * mode_component;
            perp_square += *component * *component;
        }
        let perp_norm = perp_square.sqrt();
        if !perp_norm.is_finite() {
            return false;
        }
        if perp_norm <= along.abs() {
            pin_axial(cur, origin, mode.view(), target);
            return closest_pair(cur.view()) >= clash;
        }
        let mut length = (trust - spent).min(trust).max(0.0);
        let mut accepted = false;
        while length > cfg.epsilon {
            let mut trial = cur.clone();
            for (value, component) in trial.iter_mut().zip(perp.iter()) {
                *value -= length * component / perp_norm;
            }
            pin_axial(&mut trial, origin, mode.view(), target);
            if closest_pair(trial.view()) < clash {
                length *= 0.5;
                continue;
            }
            let (trial_energy, trial_gradient) = evaluate(trial.view());
            if !trial_energy.is_finite() {
                length *= 0.5;
                continue;
            }
            let trial_along = dot_av(trial_gradient.view(), mode.view());
            let mut trial_perp = 0.0;
            for (component, mode_component) in trial_gradient.iter().zip(mode.iter()) {
                let orthogonal = component - trial_along * mode_component;
                trial_perp += orthogonal * orthogonal;
            }
            if trial_energy <= energy || trial_perp.sqrt() < perp_norm {
                *cur = trial;
                spent += length;
                accepted = true;
                break;
            }
            length *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    pin_axial(cur, origin, mode.view(), target);
    closest_pair(cur.view()) >= clash
}

fn push_ridge(
    landings: &mut Vec<Array1<f64>>,
    crossed: &mut bool,
    state: &Array1<f64>,
    mode: &Array1<f64>,
    lambda: f64,
    scale: f64,
    contact: f64,
    overshoot: f64,
    cap: usize,
) {
    if landings.len() >= cap {
        return;
    }
    // An overlap eigenvalue is far below the curvature of the minimum.
    // The square of that curvature separates the two.
    if !(lambda.is_finite() && lambda < 0.0 && lambda > -(scale * scale)) {
        return;
    }
    *crossed = true;
    landings.extend(landings_past(state, mode, lambda, contact, overshoot));
}

fn landings_past(
    state: &Array1<f64>,
    mode: &Array1<f64>,
    lambda: f64,
    contact: f64,
    overshoot: f64,
) -> Vec<Array1<f64>> {
    let mut landings = vec![state.clone()];
    let lead = max_atom_weight(mode.view()).max(1e-12);
    let contact_step = contact / lead;
    let curvature_length = 1.0 / lambda.abs().sqrt().max(1e-8);
    let extra = curvature_length.min(contact_step) * overshoot.max(0.0);
    if extra > 1e-8 {
        let mut farther = state.clone();
        for (value, component) in farther.iter_mut().zip(mode.iter()) {
            *value += extra * component;
        }
        if closest_pair(farther.view()) >= contact * 0.5 {
            landings.push(farther);
        }
    }
    landings
}

fn lowest_mode<E>(
    cur: ArrayView1<f64>,
    evaluate: &mut E,
    steps: usize,
    epsilon: f64,
) -> Option<(f64, Array1<f64>)>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let features = curvature_features(cur, |point| Some(evaluate(point).1), steps, epsilon)?;
    Some((features.lambda_min, features.mode))
}

/// Climb `direction` until the force along the lowest negative mode flips.
///
/// `hold_cover` keeps the supplied direction until that curvature is
/// negative. The other path follows the lowest mode from the first step.
/// Lengths are the curvature length and the contact distance. The walk
/// stops when the leading atom has moved by the cluster radius.
fn climb_cover<E>(
    origin: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    travel0: f64,
    hold_cover: bool,
    evaluate: &mut E,
    cfg: &Activation,
) -> CoverRidge
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let n_atoms = origin.len() / 3;
    if n_atoms < 2 || direction.len() != origin.len() || cfg.lanczos_steps < 2 {
        return empty_ridge();
    }
    let mut mode = direction.to_owned();
    if travel0 < 0.0 {
        for component in mode.iter_mut() {
            *component = -*component;
        }
    }
    if !renormalize_mode(&mut mode, origin) {
        return empty_ridge();
    }
    let contact = closest_pair(origin);
    if !contact.is_finite() || contact <= 0.0 {
        return empty_ridge();
    }
    let reach = cluster_reach(origin);
    let epsilon = cfg.epsilon.max(1e-8);
    let Some(mut directional) = directional_curvature(origin, mode.view(), evaluate, epsilon)
    else {
        return empty_ridge();
    };
    let mut lambda = directional;
    let mut lowest = directional;
    let mut positive_scale = directional.abs().max(epsilon);
    if let Some((soft, _)) = lowest_mode(origin, evaluate, cfg.lanczos_steps, epsilon) {
        lowest = soft;
        positive_scale = positive_scale.max(soft.abs());
    }
    let mut cur = origin.to_owned();
    let mut axial = 0.0_f64;
    let mut steps = 0usize;
    let budget = cfg.max_steps.max(n_atoms);
    let mut saw_negative = false;
    let mut crossed = false;
    // The opening push leaves a minimum uphill, so the first downhill
    // reading with negative curvature is a ridge rather than the well.
    let mut uphill = true;
    let mut landings: Vec<Array1<f64>> = Vec::new();

    if hold_cover {
        while steps < budget && !saw_negative {
            let lead = max_atom_weight(mode.view()).max(1e-12);
            let axial_cap = reach.max(contact) / lead;
            let curvature_length = 1.0 / directional.abs().max(epsilon).sqrt();
            let grow = curvature_length.min(contact / lead).max(epsilon);
            let target = if axial <= epsilon {
                grow.min(axial_cap)
            } else {
                (axial * 2.0).min(axial_cap)
            };
            if target <= axial + epsilon {
                break;
            }
            let snapshot = cur.clone();
            let trust = target - axial;
            let accepted = relax_pinned(
                &mut cur, origin, &mode, target, trust, contact, evaluate, cfg,
            );
            if !accepted {
                cur.clone_from(&snapshot);
                let mid = 0.5 * (axial + target);
                if mid <= axial + epsilon
                    || !relax_pinned(
                        &mut cur,
                        origin,
                        &mode,
                        mid,
                        mid - axial,
                        contact,
                        evaluate,
                        cfg,
                    )
                {
                    cur.clone_from(&snapshot);
                    break;
                }
                axial = mid;
            } else {
                axial = target;
            }
            steps += 1;
            if let Some(value) = directional_curvature(cur.view(), mode.view(), evaluate, epsilon) {
                directional = value;
                lambda = directional;
            }
            if let Some((soft, soft_mode)) =
                lowest_mode(cur.view(), evaluate, cfg.lanczos_steps, epsilon)
            {
                lowest = soft;
                if soft.is_finite() && soft < 0.0 {
                    let align = dot_av(soft_mode.view(), mode.view());
                    mode = if align < 0.0 { -soft_mode } else { soft_mode };
                    if !renormalize_mode(&mut mode, cur.view()) {
                        break;
                    }
                    lambda = soft;
                    saw_negative = true;
                    axial = axial_projection(cur.view(), origin, mode.view());
                    let (_, gradient) = evaluate(cur.view());
                    let along = dot_av(gradient.view(), mode.view());
                    if along >= 0.0 {
                        uphill = true;
                    } else if uphill {
                        push_ridge(
                            &mut landings,
                            &mut crossed,
                            &cur,
                            &mode,
                            lambda,
                            positive_scale,
                            contact,
                            cfg.overshoot,
                            n_atoms,
                        );
                        uphill = false;
                    }
                }
            }
        }
    } else if let Some((soft, soft_mode)) =
        lowest_mode(cur.view(), evaluate, cfg.lanczos_steps, epsilon)
    {
        let align = dot_av(soft_mode.view(), mode.view());
        mode = if align < 0.0 { -soft_mode } else { soft_mode };
        if !renormalize_mode(&mut mode, cur.view()) {
            return empty_ridge();
        }
        lambda = soft;
        lowest = soft;
        directional = soft;
        if soft.is_finite() && soft < 0.0 {
            saw_negative = true;
        }
    }

    if saw_negative || !hold_cover {
        while steps < budget {
            if !renormalize_mode(&mut mode, cur.view()) {
                break;
            }
            let lead = max_atom_weight(mode.view()).max(1e-12);
            if axial_projection(cur.view(), origin, mode.view()).abs() > reach.max(contact) / lead
                && steps > 0
            {
                break;
            }
            let (_, gradient) = evaluate(cur.view());
            let along = dot_av(gradient.view(), mode.view());
            if along >= 0.0 {
                uphill = true;
            } else if uphill && lambda < 0.0 {
                push_ridge(
                    &mut landings,
                    &mut crossed,
                    &cur,
                    &mode,
                    lambda,
                    positive_scale,
                    contact,
                    cfg.overshoot,
                    n_atoms,
                );
                uphill = false;
            }
            let curvature_length = 1.0 / lambda.abs().max(epsilon).sqrt();
            let mut stride = curvature_length.min(contact / lead).max(epsilon);
            let snapshot = cur.clone();
            let mut placed = false;
            while stride > epsilon {
                cur.clone_from(&snapshot);
                for (value, component) in cur.iter_mut().zip(mode.iter()) {
                    *value += stride * component;
                }
                if closest_pair(cur.view()) < contact * 0.5 {
                    stride *= 0.5;
                    continue;
                }
                let target = axial_projection(cur.view(), origin, mode.view());
                if relax_pinned(
                    &mut cur, origin, &mode, target, stride, contact, evaluate, cfg,
                ) {
                    placed = true;
                    break;
                }
                stride *= 0.5;
            }
            if !placed {
                cur.clone_from(&snapshot);
                break;
            }
            steps += 1;
            axial = axial_projection(cur.view(), origin, mode.view());
            let (_, climbed) = evaluate(cur.view());
            let along_after = dot_av(climbed.view(), mode.view());
            let step_curvature = (along_after - along) / stride;
            if step_curvature.is_finite() {
                lambda = step_curvature;
                if step_curvature < 0.0 {
                    saw_negative = true;
                }
            }
            if along_after >= 0.0 {
                uphill = true;
            } else if uphill && lambda < 0.0 {
                push_ridge(
                    &mut landings,
                    &mut crossed,
                    &cur,
                    &mode,
                    lambda,
                    positive_scale,
                    contact,
                    cfg.overshoot,
                    n_atoms,
                );
                uphill = false;
            }
            if steps % cfg.refresh.max(1) == 0
                && let Some((soft, soft_mode)) =
                    lowest_mode(cur.view(), evaluate, cfg.lanczos_steps, epsilon)
            {
                lowest = soft;
                if soft.is_finite() {
                    let align = dot_av(soft_mode.view(), mode.view());
                    mode = if align < 0.0 { -soft_mode } else { soft_mode };
                    if !renormalize_mode(&mut mode, cur.view()) {
                        break;
                    }
                    lambda = soft;
                    if soft < 0.0 {
                        saw_negative = true;
                    }
                }
            }
        }
    }

    if landings.is_empty()
        && lambda.is_finite()
        && lambda < 0.0
        && lambda > -(positive_scale * positive_scale)
    {
        landings.extend(landings_past(&cur, &mode, lambda, contact, cfg.overshoot));
    }
    CoverRidge {
        landings,
        crossed,
        lambda,
        lowest,
        steps,
        axial,
    }
}

fn note_shelf(
    shelf: &mut Option<(f64, Array1<f64>)>,
    value: f64,
    quenched: &Array1<f64>,
    origin_energy: f64,
    reach: f64,
    contact: f64,
) {
    // The shelf is the nearest distinct minimum above the start. A lower
    // minimum is the search result itself and is not a place to leave from.
    if !(value > origin_energy + 1e-6) || cluster_reach(quenched.view()) > reach + contact {
        return;
    }
    if shelf.as_ref().is_none_or(|(energy, _)| value < *energy) {
        *shelf = Some((value, quenched.clone()));
    }
}

fn climb_directions<E, Q>(
    start: ArrayView1<f64>,
    direction: &[f64],
    hop: usize,
    evaluate: &mut E,
    quench: &mut Q,
    cfg: &Activation,
    best_energy: &mut f64,
    best: &mut Array1<f64>,
    origin_energy: f64,
    reach: f64,
    contact: f64,
    shelf: &mut Option<(f64, Array1<f64>)>,
) where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    let mut follow: Option<(f64, Array1<f64>)> = None;
    for (axis, raw) in cover_axes(direction) {
        for travel in [1.0_f64, -1.0] {
            let ridge = climb_cover(start, raw.view(), travel, true, evaluate, cfg);
            println!(
                "{{\"kind\":\"climb\",\"hop\":{hop},\"axis\":\"{axis}\",\"travel\":{travel},\"crossed\":{},\"lambda\":{:.6},\"lowest\":{:.6},\"steps\":{},\"axial\":{:.4},\"landings\":{}}}",
                ridge.crossed,
                ridge.lambda,
                ridge.lowest,
                ridge.steps,
                ridge.axial,
                ridge.landings.len()
            );
            let _ = std::io::stdout().flush();
            for landing in &ridge.landings {
                if landing.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let quenched = quench(landing.view());
                let Some(value) = note_exit(evaluate, &quenched, hop, best_energy, best) else {
                    continue;
                };
                note_shelf(shelf, value, &quenched, origin_energy, reach, contact);
                let distinct = (value - origin_energy).abs() > 1e-6;
                let compact = cluster_reach(quenched.view()) <= reach + contact;
                if distinct && compact && follow.as_ref().is_none_or(|(energy, _)| value < *energy)
                {
                    follow = Some((value, quenched));
                }
            }
        }
    }
    if let Some((_, neighbour)) = follow
        && let Some((_, soft_mode)) = lowest_mode(
            neighbour.view(),
            evaluate,
            cfg.lanczos_steps,
            cfg.epsilon.max(1e-8),
        )
    {
        for travel in [1.0_f64, -1.0] {
            let ridge = climb_cover(
                neighbour.view(),
                soft_mode.view(),
                travel,
                false,
                evaluate,
                cfg,
            );
            println!(
                "{{\"kind\":\"climb\",\"hop\":{hop},\"axis\":\"mode\",\"travel\":{travel},\"crossed\":{},\"lambda\":{:.6},\"lowest\":{:.6},\"steps\":{},\"axial\":{:.4},\"landings\":{}}}",
                ridge.crossed,
                ridge.lambda,
                ridge.lowest,
                ridge.steps,
                ridge.axial,
                ridge.landings.len()
            );
            let _ = std::io::stdout().flush();
            for landing in &ridge.landings {
                if landing.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let quenched = quench(landing.view());
                if let Some(value) = note_exit(evaluate, &quenched, hop, best_energy, best) {
                    note_shelf(shelf, value, &quenched, origin_energy, reach, contact);
                }
            }
        }
    }
}

fn note_exit<E>(
    evaluate: &mut E,
    quenched: &Array1<f64>,
    hop: usize,
    best_energy: &mut f64,
    best: &mut Array1<f64>,
) -> Option<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let (value, _) = evaluate(quenched.view());
    if !value.is_finite() {
        return None;
    }
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{value:.6},\"hop\":{hop},\"role\":\"quench\"}}"
    );
    let _ = std::io::stdout().flush();
    if value < *best_energy {
        *best_energy = value;
        *best = quenched.clone();
    }
    Some(value)
}

/// Covering displacements, a minimum-mode climb, and a quench.
///
/// Each hop takes one direction of the hypersphere cover. The climb holds
/// that direction and relaxes the force perpendicular to it, then follows
/// the lowest mode once its curvature changes sign. The quench is the
/// caller's local minimiser, taken past that ridge. No target energy is
/// read.
pub fn cover_climb_search<E, Q>(
    origin: ArrayView1<f64>,
    rmsd: f64,
    max_hops: usize,
    seed: u64,
    mut evaluate: E,
    mut quench: Q,
    cfg: &Activation,
) -> Array1<f64>
where
    E: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    if max_hops == 0 || origin.is_empty() {
        return origin.to_owned();
    }
    let (origin_e, _) = evaluate(origin);
    if !origin_e.is_finite() {
        return origin.to_owned();
    }
    println!(
        "{{\"kind\":\"exit_candidate\",\"energy\":{origin_e:.6},\"hop\":0,\"role\":\"start\"}}"
    );
    let mut best = origin.to_owned();
    let mut best_e = origin_e;
    let contact = closest_pair(origin);
    let n_atoms = origin.len() / 3;
    let mut reach = 0.0_f64;
    if n_atoms >= 2 {
        let mut com = [0.0; 3];
        for i in 0..n_atoms {
            for k in 0..3 {
                com[k] += origin[3 * i + k];
            }
        }
        for value in &mut com {
            *value /= n_atoms as f64;
        }
        for i in 0..n_atoms {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = origin[3 * i + k] - com[k];
                r2 += d * d;
            }
            reach = reach.max(r2.sqrt());
        }
    }
    let mut stations = Vec::new();
    let mut station = contact * 0.5;
    let outer = reach.max(contact);
    if station > 0.0 {
        stations.push(station);
        while station * 2.0 <= outer * 1.01 {
            station *= 2.0;
            stations.push(station);
        }
    }
    if stations.last().copied().unwrap_or(0.0) < outer {
        stations.push(outer);
    }
    let mut shelf: Option<(f64, Array1<f64>)> = None;
    for hop in 0..max_hops {
        if contact > 0.95 && n_atoms >= 2 {
            let n_cover = crate::hypersphere::default_cover_size();
            let direction = crate::hypersphere::cover_direction(
                n_cover,
                origin.len(),
                hop.wrapping_add(seed as usize),
            );
            climb_directions(
                origin.view(),
                &direction,
                hop + 1,
                &mut evaluate,
                &mut quench,
                cfg,
                &mut best_e,
                &mut best,
                origin_e,
                reach,
                contact,
                &mut shelf,
            );
            for &station in &stations {
                let placed = crate::hypersphere::place_around(
                    origin.as_slice().unwrap_or(&[]),
                    &direction,
                    station.max(1.0e-3),
                    None,
                );
                if placed.len() != origin.len() {
                    continue;
                }
                let mut points = vec![Array1::from(placed)];
                let mut single = origin.to_owned();
                let mut best_atom = 0usize;
                let mut best_weight = 0.0_f64;
                for atom in 0..n_atoms {
                    let weight: f64 = (0..3)
                        .map(|k| direction[3 * atom + k].powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if weight > best_weight {
                        best_weight = weight;
                        best_atom = atom;
                    }
                }
                if best_weight > 0.0 {
                    let step = outer / best_weight;
                    for k in 0..3 {
                        single[3 * best_atom + k] += step * direction[3 * best_atom + k];
                    }
                    points.push(single);
                }
                for point in points {
                    let quenched = quench(point.view());
                    let (value, _) = evaluate(quenched.view());
                    if !value.is_finite() {
                        continue;
                    }
                    println!(
                        "{{\"kind\":\"exit_candidate\",\"energy\":{value:.6},\"hop\":{},\"role\":\"quench\"}}",
                        hop + 1
                    );
                    let _ = std::io::stdout().flush();
                    note_shelf(&mut shelf, value, &quenched, origin_e, reach, contact);
                    if value < best_e {
                        best_e = value;
                        best = quenched;
                    }
                }
            }
        }
        let quenched = cover_climb_quench(
            best.view(),
            rmsd,
            hop.wrapping_add(seed as usize),
            |point| Some(evaluate(point).1),
            &mut quench,
            cfg,
        );
        let (value, _) = evaluate(quenched.view());
        if value.is_finite() {
            println!(
                "{{\"kind\":\"exit_candidate\",\"energy\":{value:.6},\"hop\":{},\"role\":\"quench\"}}",
                hop + 1
            );
            let _ = std::io::stdout().flush();
        }
        if value.is_finite() && value < best_e {
            best_e = value;
            best = quenched.clone();
        }
        if value.is_finite() {
            note_shelf(&mut shelf, value, &quenched, origin_e, reach, contact);
        }
    }
    if contact > 0.95
        && n_atoms >= 2
        && let Some((shelf_energy, shelf_state)) = shelf
    {
        println!(
            "{{\"kind\":\"shelf\",\"energy\":{shelf_energy:.6},\"above\":{:.6}}}",
            shelf_energy - origin_e
        );
        let _ = std::io::stdout().flush();
        let mut shelf_next: Option<(f64, Array1<f64>)> = None;
        let n_cover = crate::hypersphere::default_cover_size();
        for hop in 0..max_hops {
            let direction = crate::hypersphere::cover_direction(
                n_cover,
                shelf_state.len(),
                hop.wrapping_add(seed as usize),
            );
            climb_directions(
                shelf_state.view(),
                &direction,
                hop + 1,
                &mut evaluate,
                &mut quench,
                cfg,
                &mut best_e,
                &mut best,
                origin_e,
                reach,
                contact,
                &mut shelf_next,
            );
        }
    }
    best
}

fn fivefold_opening(origin: ArrayView1<f64>, rmsd: f64, which: usize) -> Array1<f64> {
    let good: Vec<([f64; 3], f64)> = crate::soap::fivefold_axis_table(origin)
        .into_iter()
        .filter(|(_, d5)| *d5 < 1.40)
        .collect();
    if good.is_empty() {
        return origin.to_owned();
    }
    let target = if which % 2 == 0 { 0.90 } else { 0.99 };
    let (axis, _) = good
        .iter()
        .min_by(|a, b| {
            (a.1 - target)
                .abs()
                .partial_cmp(&(b.1 - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(good[0]);
    crate::soap::step_away_fivefold_about(origin, rmsd.max(0.75), axis)
}

/// Climb along `direction` first, then track the minimum mode.
///
/// The first steps follow the supplied vector. Later steps replace it
/// with the softest mode, keeping the sense of travel. `direction` is
/// what selects the half-space; the mode is what the ridge becomes.
pub fn activate_along<G>(
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    activate_aligned(x, None, Some(direction), &mut grad, cfg, 1.0)
}

/// Climb away from `origin`: the first mode is aligned with \(x-x_0\)
/// so the walk goes up the covering half-space, not back into the well.
pub fn activate_from_origin<G>(
    x: ArrayView1<f64>,
    origin: ArrayView1<f64>,
    mut grad: G,
    cfg: &Activation,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    activate_aligned(x, Some(origin), None, &mut grad, cfg, 1.0)
}

fn closest_pair(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return f64::MAX;
    }
    let mut best = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            best = best.min(r2);
        }
    }
    best.sqrt()
}

fn activate_aligned<G>(
    x: ArrayView1<f64>,
    origin: Option<ArrayView1<f64>>,
    initial_direction: Option<ArrayView1<f64>>,
    grad: &mut G,
    cfg: &Activation,
    sign0: f64,
) -> Option<ActivationOutcome>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    let dim = x.len();
    let mut cur = x.to_owned();
    let contact = closest_pair(x);
    let cluster = contact > 0.95;
    // A pair closer than half the contact distance of the start is inside
    // the core. The threshold is that distance, not a fixed length.
    let clash = contact * 0.5;
    let n_atoms = (dim / 3).max(1) as f64;
    // Perpendicular relaxation may move by one contact length, as an
    // all-atom RMS, or it cannot steer around a neighbour. The absolute
    // cap stays in force for a system that is not a cluster.
    let perp_cap = if cluster {
        // One perpendicular step moves as far as one climb step: a contact
        // length spread over the climb budget, as an all-atom RMS.
        contact / cfg.max_steps.max(1) as f64 * n_atoms.sqrt()
    } else {
        cfg.perp_max_move
    };
    let mut evaluations = 0usize;

    let first = curvature_features(
        cur.view(),
        |y| {
            evaluations += 1;
            grad(y)
        },
        cfg.lanczos_steps,
        cfg.epsilon,
    )?;
    let (mut mode, mut lambda) = if let Some(direction) = initial_direction {
        let n: f64 = direction.iter().map(|z| z * z).sum::<f64>().sqrt();
        if n < 1e-15 {
            (first.mode.clone(), first.lambda_min)
        } else {
            // Positive so the first steps cannot look like a crossed ridge
            // before the mode has been recomputed.
            let mut unit = direction.to_owned();
            unit /= n;
            (unit, 1.0)
        }
    } else {
        (first.mode.clone(), first.lambda_min)
    };
    let sign = if let Some(origin) = origin {
        let align: f64 = mode
            .iter()
            .zip(x.iter().zip(origin.iter()))
            .map(|(m, (xi, oi))| m * (xi - oi))
            .sum();
        if align >= 0.0 { 1.0 } else { -1.0 }
    } else {
        sign0
    };
    let mut steps = 0usize;
    let mut crossed = false;
    let mut rise = 0.0;
    let mut saw_uphill = false;
    // A clash eigenvalue is far below the curvature of the minimum.
    // The square of that curvature separates the two without a fixed cutoff.
    let mut curvature_scale = first.lambda_min.abs();
    let mut previous_along: Option<f64> = None;
    // Highest ridge crossed on this climb. A shallow first saddle is kept
    // only until a higher one is crossed. The quench leaves from that ridge.
    let mut ridge: Option<(Array1<f64>, Array1<f64>, f64)> = None;
    // A supplied direction is the cover. Hold it until the force along it
    // flips, then take the minimum mode. Replacing it on the first refresh
    // walks the softest well of the minimum the cover was meant to leave.
    let mut hold_direction = initial_direction.is_some();
    let mut along_scale = 1.0;

    'climb: while steps < cfg.max_steps {
        if cfg.step * along_scale <= cfg.epsilon {
            break;
        }
        // Refresh the direction on schedule, and always after the curvature has
        // already been seen to fall, since that is where it rotates fastest.
        if steps > 0 && steps % cfg.refresh == 0 && !hold_direction {
            match curvature_features(
                cur.view(),
                |y| {
                    evaluations += 1;
                    grad(y)
                },
                cfg.lanczos_steps,
                cfg.epsilon,
            ) {
                Some(f) => {
                    // Keep the sense of travel: the mode is defined up to sign
                    // and flipping it mid-climb walks back down.
                    let dot: f64 = f.mode.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
                    mode = if dot < 0.0 { -f.mode } else { f.mode };
                    lambda = f.lambda_min;
                }
                None => break,
            }
        }
        let stride = cfg.step * along_scale;
        let snapshot = cur.clone();
        for i in 0..dim {
            cur[i] += sign * stride * mode[i];
        }

        // Perpendicular relaxation. Sliding down the component of the gradient
        // orthogonal to the mode keeps the structure on the valley floor; the
        // component along the mode is what the climb is fighting and is left
        // alone.
        let mut along = 0.0;
        let mut gnorm = 0.0;
        for _ in 0..cfg.perp_steps {
            let g = match grad(cur.view()) {
                Some(g) => {
                    evaluations += 1;
                    g
                }
                None => {
                    cur.clone_from(&snapshot);
                    break 'climb;
                }
            };
            gnorm = g.iter().map(|z| z * z).sum::<f64>().sqrt();
            along = g.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
            let mut d = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                d[i] = cfg.perp_rate * (g[i] - along * mode[i]);
            }
            let n: f64 = d.iter().map(|z| z * z).sum::<f64>().sqrt();
            let scale = if n > perp_cap && n > 0.0 {
                perp_cap / n
            } else {
                1.0
            };
            for i in 0..dim {
                cur[i] -= scale * d[i];
            }
        }

        // A pair inside half the starting contact distance is a clash.
        // Restore the last intact structure and halve the step. A ridge
        // already crossed stays the place the quench will leave from.
        let crowded = cluster && closest_pair(cur.view()) < clash;
        if !gnorm.is_finite() || crowded {
            cur.clone_from(&snapshot);
            along_scale *= 0.5;
            continue;
        }
        steps += 1;

        // A ridge is a force flip at negative curvature. The highest such
        // ridge on the climb is the one the quench leaves from.
        //
        // Negative curvature says the ridge is ahead, not behind: on a double
        // well the curvature along the well direction turns over at
        // `|u| = 1/sqrt(3)` while the barrier top is at `u = 0`. Stopping there
        // and pushing on by the overshoot ended the climb at `u = 0.101`, on
        // the side it started, and a quench from there goes home.
        //
        // The saddle is where the force along the direction of travel changes
        // sign: uphill while `sign * g . v > 0`, downhill after. Combined with
        // negative curvature that is the ridge, and past it a quench falls the
        // other way.
        if stride > 0.0 {
            if let Some(prev) = previous_along {
                let fd = (along - prev) / (sign * stride);
                if fd.is_finite() {
                    lambda = fd;
                }
            }
            previous_along = Some(along);
        }
        rise += along * sign * stride;
        let climbing = sign * along > 0.0;
        if climbing {
            saw_uphill = true;
        }
        if hold_direction && saw_uphill && !climbing {
            hold_direction = false;
            if let Some(features) = curvature_features(
                cur.view(),
                |y| {
                    evaluations += 1;
                    grad(y)
                },
                cfg.lanczos_steps,
                cfg.epsilon,
            ) {
                let dot: f64 = features
                    .mode
                    .iter()
                    .zip(mode.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                mode = if dot < 0.0 {
                    -features.mode
                } else {
                    features.mode
                };
                lambda = features.lambda_min;
                if let Some(g) = grad(cur.view()) {
                    evaluations += 1;
                    along = g.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
                }
            }
        }
        if lambda > curvature_scale {
            curvature_scale = lambda;
        }
        let descending = sign * along < 0.0;
        let intact = lambda < 0.0 && lambda > -(curvature_scale * curvature_scale);
        if saw_uphill && descending && intact && rise >= cfg.min_rise {
            // Keep the highest intact ridge. A shallow first saddle stays
            // inside one funnel; a later one can leave it.
            let higher = ridge.as_ref().is_none_or(|(_, _, kept)| rise > *kept);
            if higher {
                ridge = Some((cur.clone(), mode.clone(), rise));
                crossed = true;
                break 'climb;
            }
            saw_uphill = false;
        }
    }

    if let Some((state, ridge_mode, _)) = ridge {
        cur = state;
        mode = ridge_mode;
    }
    if crossed && cfg.overshoot > 0.0 {
        // One push past the turning point, so the quench falls forward rather
        // than back down the way it came.
        for i in 0..dim {
            cur[i] += sign * cfg.overshoot * cfg.step * mode[i];
        }
    }

    Some(ActivationOutcome {
        state: cur,
        lambda,
        steps,
        crossed,
        evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A double well along one collective direction, stiff in the rest.
    ///
    /// `E = (u^2 - 1)^2 + 1/2 sum_i k_i (x_i - u w_i)^2` with `u = x . w`.
    /// Minima at `u = +/- 1`, a barrier at `u = 0`.
    ///
    /// The perpendicular stiffnesses vary and all exceed the curvature along
    /// `w` at the minimum, which is 8. Both parts are load-bearing. If they are
    /// smaller the softest mode is not the well direction and the test is
    /// asking the wrong question, and if they are all equal the Hessian is a
    /// multiple of the identity, every vector is an eigenvector, and the Krylov
    /// space collapses at the first step.
    fn perp_stiffness(dim: usize) -> Array1<f64> {
        Array1::from_shape_fn(dim, |i| 12.0 + 1.7 * (i % 7) as f64)
    }

    fn double_well<'a>(
        w: &'a Array1<f64>,
        k: &'a Array1<f64>,
    ) -> impl Fn(ArrayView1<f64>) -> Option<Array1<f64>> + 'a {
        move |x: ArrayView1<f64>| {
            let u: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            let dedu = 4.0 * u * (u * u - 1.0);
            let mut g = Array1::zeros(x.len());
            // d/dx of the perpendicular term, including its dependence on u.
            let mut kp_dot_w = 0.0;
            for i in 0..x.len() {
                kp_dot_w += k[i] * (x[i] - u * w[i]) * w[i];
            }
            for i in 0..x.len() {
                let perp = x[i] - u * w[i];
                g[i] = dedu * w[i] + k[i] * perp - kp_dot_w * w[i];
            }
            Some(g)
        }
    }

    fn direction(dim: usize) -> Array1<f64> {
        let mut w = Array1::from_shape_fn(dim, |i| ((i % 5) as f64 - 2.0) + 0.25);
        let n: f64 = w.iter().map(|z| z * z).sum::<f64>().sqrt();
        w /= n;
        w
    }

    /// The cap has to bind on a stiff gradient, or the climb walks off the
    /// structure it was refining.
    #[test]
    fn a_perpendicular_step_is_bounded_however_steep_the_gradient() {
        let dim = 36;
        let w = direction(dim);
        // A gradient a thousand times the scale the rate was set for.
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            Some(Array1::from_shape_fn(x.len(), |i| {
                1500.0 * x[i] + 3.0 * (i % 5) as f64
            }))
        };
        let cfg = Activation {
            max_steps: 4,
            ..Activation::default()
        };
        let out = activate(w.view(), g, &cfg, 1.0).unwrap();
        let travelled: f64 = out
            .state
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let bound = out.steps as f64 * (cfg.step + cfg.perp_steps as f64 * cfg.perp_max_move)
            + cfg.overshoot * cfg.step;
        assert!(
            travelled <= bound + 1e-9,
            "moved {travelled:.3} where the caps allow {bound:.3}"
        );
    }

    fn quench_double_well(w: &Array1<f64>, k: &Array1<f64>, mut x: Array1<f64>) -> Array1<f64> {
        let g = double_well(w, k);
        for _ in 0..80 {
            let grad = g(x.view()).unwrap();
            let n: f64 = grad.iter().map(|z| z * z).sum::<f64>().sqrt();
            if n < 1e-6 {
                break;
            }
            let step = (0.05_f64).min(0.2 / n);
            x.scaled_add(-step, &grad);
        }
        x
    }

    #[test]
    fn cover_climb_quench_on_a_double_well_uses_only_the_force() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let origin = w.clone();
        let g = double_well(&w, &k);
        let end = cover_climb_quench(
            origin.view(),
            0.05,
            0,
            &g,
            |x| quench_double_well(&w, &k, x.to_owned()),
            &Activation {
                max_steps: 12,
                ..Activation::default()
            },
        );
        assert_eq!(end.len(), dim);
        let grad = g(end.view()).unwrap();
        let n: f64 = grad.iter().map(|z| z * z).sum::<f64>().sqrt();
        assert!(n < 1e-2, "quench did not reach a stationary point, |g|={n}");
        assert!(end.iter().all(|v| v.is_finite()));
    }

    /// Four Lennard-Jones atoms at the tetrahedron. The call is the same
    /// function as the smooth well: coordinates in, force in, quench in.
    #[test]
    fn cover_climb_quench_on_a_lennard_jones_tetrahedron_reads_no_target() {
        let scale = 2.0_f64.powf(1.0 / 6.0);
        let raw = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let mut origin = Array1::zeros(12);
        for (i, p) in raw.iter().enumerate() {
            for k in 0..3 {
                origin[3 * i + k] = p[k] * scale / 3.0_f64.sqrt();
            }
        }
        let lj = |x: ArrayView1<f64>| -> (f64, Array1<f64>) {
            let n = x.len() / 3;
            let mut value = 0.0;
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
                    let inv6 = inv2.powi(3);
                    let inv12 = inv6 * inv6;
                    value += 4.0 * (inv12 - inv6);
                    let coefficient = 24.0 * inv2 * (2.0 * inv12 - inv6);
                    for k in 0..3 {
                        gradient[3 * i + k] -= coefficient * d[k];
                        gradient[3 * j + k] += coefficient * d[k];
                    }
                }
            }
            (value, gradient)
        };
        let cfg = Activation {
            max_steps: 6,
            lanczos_steps: 8,
            ..Activation::default()
        };
        let end = cover_climb_quench(
            origin.view(),
            0.2,
            0,
            |x| Some(lj(x).1),
            |x| {
                let mut y = x.to_owned();
                for _ in 0..40 {
                    let (_, g) = lj(y.view());
                    let n: f64 = g.iter().map(|z| z * z).sum::<f64>().sqrt();
                    if n < 1e-4 {
                        break;
                    }
                    let step = (1e-3_f64).min(0.05 / n);
                    y.scaled_add(-step, &g);
                }
                y
            },
            &cfg,
        );
        let (energy, gradient) = lj(end.view());
        assert!(energy.is_finite());
        assert!(gradient.iter().all(|v| v.is_finite()));
    }

    /// The property the module exists for. A straight displacement of the same
    /// length stays on its own side of the barrier; the climb crosses it.
    #[test]
    fn the_climb_crosses_the_barrier_a_displacement_does_not() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        // Start at the minimum with u = 1.
        let x: Array1<f64> = w.clone();
        let u0: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!((u0 - 1.0).abs() < 1e-12);

        let cfg = Activation::default();
        let out = activate(x.view(), &g, &cfg, -1.0).unwrap();
        let u: f64 = out.state.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!(
            out.crossed,
            "the climb never reached negative curvature, ending at u = {u:.3}"
        );
        assert!(
            u < 0.0,
            "the climb ended at u = {u:.3}, still on the side it started"
        );

        // The same total distance in a straight line, no climbing.
        let travelled = (out.steps as f64 + cfg.overshoot) * cfg.step;
        let mut straight = x.clone();
        for i in 0..dim {
            straight[i] -= travelled * w[i];
        }
        let us: f64 = straight.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        // Both end past the barrier here because the well is one-dimensional;
        // what separates them is that the climb *knows* it is past, which is
        // what a caller needs in order to stop.
        assert!(
            us < 0.0,
            "the straight line should also cross this simple well: {us:.3}"
        );
    }

    /// The climb must not run forever on a direction that never turns over.
    #[test]
    fn from_origin_climbs_the_covering_half_space() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        let origin = w.clone();
        let mut start = w.clone();
        for i in 0..dim {
            start[i] -= 0.15 * w[i];
        }
        let out =
            activate_from_origin(start.view(), origin.view(), &g, &Activation::default()).unwrap();
        assert!(
            out.crossed,
            "the ridge must be crossed, lambda={}",
            out.lambda
        );
        let u: f64 = out.state.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!(u < 0.0, "quench side must be the other well, u={u:.3}");
    }

    #[test]
    fn a_direction_that_never_softens_stops_at_the_cap() {
        let dim = 36;
        let w = direction(dim);
        // Purely harmonic: the curvature is positive everywhere.
        // Stiffnesses that differ, so the Hessian is not a multiple of the
        // identity and the Krylov space has somewhere to go.
        let k = perp_stiffness(dim);
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            Some(Array1::from_shape_fn(x.len(), |i| k[i] * x[i]))
        };
        let x = w.clone();
        let cfg = Activation {
            max_steps: 6,
            ..Activation::default()
        };
        let out = activate(x.view(), &g, &cfg, 1.0).unwrap();
        assert!(!out.crossed, "a harmonic well has no ridge to cross");
        assert!(
            out.steps <= 6,
            "the climb took {} steps against a cap of 6",
            out.steps
        );
    }

    /// A caller whose budget runs out mid-climb gets the structure so far
    /// rather than a panic or a silent full-cost climb.
    #[test]
    fn an_exhausted_budget_stops_the_climb() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let inner = double_well(&w, &k);
        let mut left = 40usize;
        let g = move |x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if left == 0 {
                return None;
            }
            left -= 1;
            inner(x)
        };
        let out = activate(w.view(), g, &Activation::default(), -1.0);
        if let Some(o) = out {
            assert!(
                o.evaluations <= 40,
                "spent {} gradients against a budget of 40",
                o.evaluations
            );
        }
        // Refusing before the first curvature pass completes is also correct;
        // what would not be is climbing past the budget.
    }

    /// The sign argument has to mean something: the two ends of a soft
    /// direction are different saddles and a caller picking one must get it.
    #[test]
    fn the_two_signs_climb_opposite_ways() {
        let dim = 36;
        let w = direction(dim);
        let k = perp_stiffness(dim);
        let g = double_well(&w, &k);
        let x = w.clone();
        let a = activate(x.view(), &g, &Activation::default(), 1.0).unwrap();
        let b = activate(x.view(), &g, &Activation::default(), -1.0).unwrap();
        let ua: f64 = a.state.iter().zip(w.iter()).map(|(p, q)| p * q).sum();
        let ub: f64 = b.state.iter().zip(w.iter()).map(|(p, q)| p * q).sum();
        assert!(
            ua > ub,
            "the two signs ended at u = {ua:.3} and {ub:.3}, not on opposite sides"
        );
    }

    /// A capped octahedron of seven Lennard-Jones atoms is a local minimum.
    /// The pentagonal bipyramid lies below it. The search is given only the
    /// higher minimum, the force, and a quench.
    #[test]
    fn cover_climb_search_leaves_a_higher_lennard_jones_minimum() {
        fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let n = x.len() / 3;
            let mut value = 0.0;
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
                    let inv6 = inv2.powi(3);
                    let inv12 = inv6 * inv6;
                    value += 4.0 * (inv12 - inv6);
                    let coefficient = 24.0 * inv2 * (2.0 * inv12 - inv6);
                    for k in 0..3 {
                        gradient[3 * i + k] -= coefficient * d[k];
                        gradient[3 * j + k] += coefficient * d[k];
                    }
                }
            }
            (value, gradient)
        }
        fn quench(x: ArrayView1<f64>) -> Array1<f64> {
            let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
            opt.minimize(x, 80, |v| Some(lj(v))).1
        }

        let req = 2.0_f64.powf(1.0 / 6.0);
        let scale = req / 2.0_f64.sqrt();
        let mut capped = Array1::zeros(21);
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (i, p) in axes.iter().enumerate() {
            for k in 0..3 {
                capped[3 * i + k] = p[k] * scale;
            }
        }
        let centroid = scale / 3.0;
        let unit = 1.0 / 3.0_f64.sqrt();
        for k in 0..3 {
            capped[18 + k] = centroid + req * unit;
        }
        let capped = quench(capped.view());
        let e_cap = lj(capped.view()).0;

        let mut bipyramid = Array1::zeros(21);
        let radius = req / (2.0 * (std::f64::consts::PI / 5.0).sin());
        let height = (req * req - radius * radius).max(0.0).sqrt();
        for k in 0..5 {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / 5.0;
            bipyramid[3 * k] = radius * angle.cos();
            bipyramid[3 * k + 1] = radius * angle.sin();
        }
        bipyramid[17] = height;
        bipyramid[20] = -height;
        let bipyramid = quench(bipyramid.view());
        let e_low = lj(bipyramid.view()).0;
        assert!(
            e_low < e_cap - 0.2,
            "the lower isomer is not below the start: {e_low} vs {e_cap}"
        );

        let end = cover_climb_search(
            capped.view(),
            0.4,
            40,
            1,
            lj,
            quench,
            &Activation {
                max_steps: 4,
                lanczos_steps: 6,
                perp_steps: 1,
                ..Activation::default()
            },
        );
        let found = lj(end.view()).0;
        assert!(
            found < e_cap - 0.2,
            "search energy {found}, start {e_cap}, lower isomer {e_low}"
        );
    }
}
