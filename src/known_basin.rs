//! Invert basins already occupied so rgmin will not walk back into them.
//!
//! Henkelman and Jónsson, *J. Chem. Phys.* **1999**, *111*, 7010
//! <https://doi.org/10.1063/1.480097>: the dimer replaces \(F\) by
//! \(F-2(F\cdot P)P\) so a first-order stepper walks *up* the lowest
//! mode to a saddle. Occupancy Leave applies the same Householder in
//! the DECAF packing map, not in Cartesian radius. \(\mu\) is the
//! mean of per-center [`crate::soap::local_nu3_z`] (SOAP plus ACE
//! \(\nu=3\)) at [`crate::catalog::packing::PACKING_SPEC`]. For the
//! nearest archived packing \(\mu_k\),
//! \(\hat u_\varphi=(\mu-\mu_k)/\|\mu-\mu_k\|\) and
//! \(P=J_\mu^{\mathsf T}\hat u_\varphi\) is the Cartesian pullback
//! through the analytic stacked Jacobian: the same increment-on-every-center
//! lift as [`crate::soap::kick_packing_nu3`]. Then
//! \(g\leftarrow g-2(g\cdot\hat P)\hat P\) when \(g\cdot P>0\).
//! Leftover SOAP \(p_i-\mu\) collapses across the paper funnels; this
//! map does not.
//!
//! When the packing mean is unavailable (fewer than a closed-shell
//! neighbour cloud), the same Householder falls back to the COM-free
//! Cartesian radius of each well. That fallback is not a packing walk.
//!
//! The Householder on \(g\) is not conservative. The line search is
//! run on the matching PES \(E+V\), a Gaussian hill on each known
//! well in the same map. After a transformed quench that changes
//! DECAF family, a raw-\(E\) polish sits on a true minimum of that
//! well.

use std::cell::RefCell;

use ndarray::{Array1, Array2, ArrayView1};

use crate::catalog::packing::{MINIMUM_PACKING_ATOMS, PACKING_SPEC};
use crate::movekernel::MoveKernel;
use crate::soap::{jacobian_nu3, local_nu3_z};

struct Well {
    coords: Array1<f64>,
    packing_mean: Option<Array1<f64>>,
    /// \(T\ln n_k\): the entropic part of this packing's free-energy
    /// depth, from the arrivals the run has recorded on it.
    entropy: f64,
    /// Bias already standing here, for the well-tempered scaling.
    deposit: f64,
}

struct Armed {
    wells: Vec<Well>,
    sigma_rmsd: f64,
    /// Raw \(E\) at the last point the transform was asked about, so the
    /// walk can see the ridge it is crossing rather than the hill it is on.
    raw_energy: Option<f64>,
    lift: Option<f64>,
    sigma_phi: Option<f64>,
    /// Geometry the cached packing modes were lifted at.
    mode_x: Option<Array1<f64>>,
    /// Cached \((\hat P, \|P\|)\) per well, in `wells` order.
    modes: Vec<(usize, Array1<f64>, f64)>,
    /// \(\Delta T\) of the well-tempered scaling. Zero disables it and
    /// leaves every hill at full height.
    delta_t: f64,
    /// Neighbour invert lifted at arm. Hops apply this \(P\) until the
    /// next arm; they do not rebuild \(J\).
    frozen: bool,
    /// \((k,\hat P,r_\varphi,\|P\|)\) at arm, used when [`Self::frozen`].
    frozen_modes: Vec<(usize, Array1<f64>, f64, f64)>,
}

thread_local! {
    static ARMED: RefCell<Option<Armed>> = const { RefCell::new(None) };
}

/// Arm the transformed quench for one occupancy Leave.
///
/// `origin` is the live well this extra is leaving, and `references` are the
/// packings already on file, as coordinates. Every one of them contributes a
/// mode the Householder inverts, so the walk is pushed away from all of them
/// and not only out of the well it starts in. The leftover-SOAP archive is a
/// list of cloud means, not structures, so it cannot supply these: a Leave
/// that reads it sees vectors of the wrong length and inverts nothing but its
/// own well.
pub fn arm_leave(origin: ArrayView1<f64>, sigma_rmsd: f64, references: &[Vec<f64>]) {
    let book: Vec<crate::catalog::PackingReference> = references
        .iter()
        .map(|coordinates| crate::catalog::PackingReference {
            coordinates: coordinates.clone(),
            visits: 1,
            deposit: 0.0,
        })
        .collect();
    arm_leave_free(origin, sigma_rmsd, &book, 0.0, 0.0);
}

/// Arm the Leave against the free-energy depth of the known packings.
///
/// The hill on well \(k\) is its depth along the packing coordinate, and
/// a depth in potential energy is the wrong quantity to fill. What holds
/// a chain is \(F_k=E_k-TS_k\), and on a cluster the configurational
/// entropy of a packing is the log of how many ways the run reaches it,
/// which is the arrival count the reference cloud now carries. On LJ75
/// that difference decides the answer: the Marks decahedron is
/// 1.210082 \(\varepsilon\) below the icosahedral floor, but the
/// icosahedral shelf carries hundreds of minima against one narrow
/// decahedral well, so at the run temperature \(0.8\) an entropy ratio
/// of a thousand is already \(5.5\,\varepsilon\) in the other
/// direction. A chain that stays icosahedral is at the free-energy
/// minimum, and no deposit shaped like a potential energy can say
/// otherwise (Mandelshtam, Frantsuzov, Calvo, *J. Phys. Chem. A* **2006**,
/// *110*, 5326).
///
/// `delta_t` is the well-tempered scale of Barducci, Bussi and Parrinello
/// (*Phys. Rev. Lett.* **2008**, *100*, 020603): a hill on a well that
/// already carries \(V_k\) is scaled by \(e^{-V_k/\Delta T}\), so the
/// pile converges to \(-\Delta T/(T+\Delta T)\) times the free energy
/// instead of growing without bound. Zero leaves every hill at full
/// height, which is the non-tempered case of Laio and Parrinello
/// (*Proc. Natl. Acad. Sci.* **2002**, *99*, 12562).
pub fn arm_leave_free(
    origin: ArrayView1<f64>,
    sigma_rmsd: f64,
    references: &[crate::catalog::PackingReference],
    temperature: f64,
    delta_t: f64,
) {
    let mut wells = vec![Well {
        packing_mean: packing_mean(origin),
        coords: origin.to_owned(),
        entropy: 0.0,
        deposit: 0.0,
    }];
    for reference in references {
        if reference.coordinates.len() != origin.len() {
            continue;
        }
        let coords = Array1::from(reference.coordinates.clone());
        if same_point(coords.view(), origin) {
            continue;
        }
        wells.push(Well {
            packing_mean: packing_mean(coords.view()),
            coords,
            entropy: temperature.max(0.0) * f64::from(reference.visits.max(1)).ln(),
            deposit: reference.deposit.max(0.0),
        });
    }
    // The well the chain is standing on carries the entropy of the
    // packing it is leaving, which is the term that has to be paid to
    // leave it. Its own entry in the cloud holds the count; when the
    // origin is not on file yet the run has still arrived there at least
    // as often as anywhere else it has been, so the largest count stands
    // in rather than zero, which would price the funnel it is stuck in at
    // nothing.
    //
    // Its standing deposit comes from the same entry, and has to: the
    // origin is the well a chain Leaves from over and over, so it is the
    // one that accumulates bias fastest, and it is skipped by the loop
    // above as the well being left. Leaving its deposit at zero exempted
    // the only well well-tempering was there to converge.
    let own = references.iter().find(|reference| {
        if reference.coordinates.len() != origin.len() {
            return false;
        }
        let held = Array1::from(reference.coordinates.clone());
        same_point(held.view(), origin)
    });
    let origin_visits = own.map(|reference| reference.visits).or_else(|| {
        references
            .iter()
            .filter(|reference| reference.coordinates.len() == origin.len())
            .map(|reference| reference.visits)
            .max()
    });
    if let Some(first) = wells.first_mut() {
        if let Some(origin_visits) = origin_visits {
            first.entropy = temperature.max(0.0) * f64::from(origin_visits.max(1)).ln();
        }
        if let Some(own) = own {
            first.deposit = own.deposit.max(0.0);
        }
    }
    let frozen_modes = freeze_packing_modes(origin, &wells);
    ARMED.with(|slot| {
        *slot.borrow_mut() = Some(Armed {
            wells,
            raw_energy: None,
            sigma_rmsd: sigma_rmsd.max(1e-6),
            lift: None,
            sigma_phi: None,
            mode_x: Some(origin.to_owned()),
            modes: frozen_modes
                .iter()
                .map(|(index, p, _, p_norm)| (*index, p.clone(), *p_norm))
                .collect(),
            delta_t: delta_t.max(0.0),
            frozen: !frozen_modes.is_empty(),
            frozen_modes,
        });
    });
}

/// Drop the transform. Later rgmin calls see the raw PES.
pub fn disarm() {
    ARMED.with(|slot| *slot.borrow_mut() = None);
}

/// Whether a Leave quench is currently transformed.
pub fn is_armed() -> bool {
    ARMED.with(|slot| slot.borrow().is_some())
}

/// Whether hops apply the \(P\) lifted at arm, without rebuilding \(J\).
pub fn invert_is_frozen() -> bool {
    ARMED.with(|slot| slot.borrow().as_ref().is_some_and(|armed| armed.frozen))
}

/// Hill amplitude \(A\) and width \(\sigma_\varphi\) of the armed
/// transform, once the first transformed evaluation has set them.
///
/// The deposited potential is \(A\sum_k e^{-r_k^2/2\sigma_\varphi^2}\),
/// so what decides whether a wider ensemble can lift a chain out of its
/// funnel is \(A\) against the barrier it has to clear: on LJ75 the
/// icosahedral--Marks saddles sit 8.69 and 7.48 \(\varepsilon\) above the
/// shelf (Doye, Wales, Berry, *J. Chem. Phys.* **1995**, *103*, 4234).
/// Reported so a run can be read against that number rather than against
/// whether it happened to escape.
pub fn lift() -> Option<(f64, f64)> {
    ARMED.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|armed| Some((armed.lift?, armed.sigma_phi?)))
    })
}

/// Run `body` on the raw surface, then restore the armed transform.
///
/// The polish that decides where a walk landed has to run on \(E\), not on
/// \(E+V\): a hill the Leave put there is not part of the landscape, and a
/// minimum of the sum is not a minimum of the potential.
pub fn with_disarmed<T>(body: impl FnOnce() -> T) -> T {
    let held = ARMED.with(|slot| slot.borrow_mut().take());
    let out = body();
    ARMED.with(|slot| *slot.borrow_mut() = held);
    out
}

/// Suppress only the Householder while `body` runs, keeping the hill.
///
/// The reflection is a poor-man's min-mode: it inverts the force along one
/// direction chosen in advance, which is a guess about where the exit is.
/// A min-mode search does not need that guess, but it does need a force
/// that is the gradient of something, and a reflected force is not.
///
/// The hill is a different object and is wanted: filling the occupied well
/// by \(A\) lowers the saddle out of it by the same amount, so a dimer on
/// \(E+V\) looks for a barrier the deposit has already paid down. That is
/// the pairing metadynamics and saddle search are usually put in, and it
/// is not available while the two are welded together.
pub fn with_hill_only<T>(body: impl FnOnce() -> T) -> T {
    let held = FLAT.with(|slot| slot.replace(true));
    let out = body();
    FLAT.with(|slot| slot.set(held));
    out
}

thread_local! {
    /// Whether the Householder is suppressed for the current call.
    static FLAT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn householder_suppressed() -> bool {
    FLAT.with(std::cell::Cell::get)
}

/// Largest Cartesian RMSD one armed step may take.
///
/// The hill width is the size of the well being left; the walk across it is
/// not one step of that size. A trust radius equal to the width lets an
/// accepted uphill step carry atoms through each other, and the walk then
/// climbs for the whole iteration budget and reports a structure at
/// \(10^{11}\varepsilon\) that no quench recovers.
pub const LEAVE_WALK_STEP: f64 = 0.10;

/// Largest Cartesian RMSD one armed walk may cover in total.
///
/// A cooperative rearrangement moves atoms by about a nearest-neighbour
/// separation. Past that the walk is not crossing a ridge between packings,
/// it is pulling the cluster apart, and the ridge rule cannot end a climb
/// that never turns over.
pub const LEAVE_WALK_SPAN: f64 = 0.8;

/// Consecutive falls in raw \(E\) that name the far side of a ridge.
pub const LEAVE_WALK_DESCENTS: usize = 3;

/// Largest climb in raw \(E\), in units of the well depth per atom.
///
/// The span rule alone accepts a step that puts two atoms on top of each
/// other, because overlapping atoms move \(\mu\) further than any
/// rearrangement does. Wales and Doye put the LJ75 ico-Marks barriers at
/// 8.69 and 7.48 \(\varepsilon\) against a well depth of 5.28
/// \(\varepsilon\) per atom, so four times the depth per atom clears the
/// ridge the walk is meant to cross and refuses the ones it is not.
pub const LEAVE_WALK_CLIMB: f64 = 4.0;

/// rgmin step on the transformed surface: two-loop direction, accept a step
/// that increases the span from the known wells. Span is packing L2
/// \(\min_k\|\mu-\mu_k\|\) when the wells carry a \(\nu=3\) mean,
/// otherwise COM-free RMSD. Raw \(E\) may rise; that is the dimer walk away
/// from an occupied packing.
///
/// The walk ends at the ridge, the way an activation-relaxation walk does:
/// raw \(E\) climbs while the invert holds the descent component that
/// points back at the wells on file, and the first steps where it falls
/// again are the far side. Quenching from there is what decides which
/// packing the walk landed in.
///
/// Two stopping rules that do not work, both measured from the LJ75
/// icosahedral minimum. Crossing the DECAF grain stops the walk on a
/// distorted geometry that has not crossed anything: the quench that follows
/// returns to \(-396.282249\), the floor it started on, every time. Having
/// no rule at all lets the walk climb for the whole iteration budget and
/// report structures between \(10^4\) and \(10^{11}\varepsilon\), which
/// is atoms sitting on each other. [`LEAVE_WALK_SPAN`] bounds the second
/// case, since a raw energy that rises monotonically never turns over.
pub fn step_rgmin<F>(
    opt: &mut crate::methods::warm_lbfgs::WarmLbfgs,
    x0: ArrayView1<f64>,
    max_iter: usize,
    mut fg: F,
) -> (f64, Array1<f64>)
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let sigma = ARMED.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|a| a.sigma_rmsd.min(LEAVE_WALK_STEP))
            .unwrap_or(LEAVE_WALK_STEP)
    });
    let mut x = x0.to_owned();
    let Some((mut energy, mut grad)) = fg(x.view()) else {
        return (f64::INFINITY, x);
    };
    let n_at = (x.len() / 3).max(1) as f64;
    let start_raw = raw_energy();
    let ceiling = start_raw.map(|raw| raw + LEAVE_WALK_CLIMB * raw.abs() / n_at);
    let mut peak = start_raw.unwrap_or(f64::NEG_INFINITY);
    let mut descents = 0usize;
    for _ in 0..max_iter {
        let mut direction = opt.two_loop(grad.view());
        if direction.dot(&grad) >= 0.0 {
            direction = grad.mapv(|v| -v);
        }
        let step_rmsd = (direction.iter().map(|v| v * v).sum::<f64>() / n_at).sqrt();
        if step_rmsd < 1e-15 {
            break;
        }
        let mut alpha = (sigma / step_rmsd).min(1.0);
        let start_span = span_from_wells(x.view());
        let mut accepted = false;
        for _ in 0..8 {
            let mut trial = x.clone();
            trial.scaled_add(alpha, &direction);
            let Some((te, tg)) = fg(trial.view()) else {
                return (energy, x);
            };
            let trial_span = span_from_wells(trial.view());
            let climbed = match (raw_energy(), ceiling) {
                (Some(raw), Some(ceiling)) => raw > ceiling,
                _ => false,
            };
            if !climbed && (trial_span > start_span || te < energy) {
                let s = &trial - &x;
                let y = &tg - &grad;
                opt.record_pair(s, y);
                x = trial;
                energy = te;
                grad = tg;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        match raw_energy() {
            Some(raw) if raw > peak => {
                peak = raw;
                descents = 0;
            }
            Some(_) => descents += 1,
            None => {}
        }
        if descents >= LEAVE_WALK_DESCENTS {
            break;
        }
        let travelled = (x
            .iter()
            .zip(x0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n_at)
            .sqrt();
        if travelled >= LEAVE_WALK_SPAN {
            break;
        }
    }
    (energy, x)
}

/// Distance from `x` to the nearest armed well.
///
/// Packing L2 \(\min_k\|\mu-\mu_k\|\) when the wells carry a
/// \(\nu=3\) mean, otherwise COM-free RMSD. Zero when unarmed.
/// Occupancy Leave keeps the invert walk when this rises.
pub fn span(x: ArrayView1<f64>) -> f64 {
    span_from_wells(x)
}

fn span_from_wells(x: ArrayView1<f64>) -> f64 {
    ARMED.with(|slot| {
        let held = slot.borrow();
        let Some(armed) = held.as_ref() else {
            return 0.0;
        };
        if armed.wells.iter().any(|well| well.packing_mean.is_some()) {
            let Some(mu) = packing_mean(x) else {
                return 0.0;
            };
            return armed
                .wells
                .iter()
                .filter_map(|well| well.packing_mean.as_ref())
                .map(|mu_k| packing_l2(mu.view(), mu_k.view()))
                .fold(f64::INFINITY, f64::min);
        }
        armed
            .wells
            .iter()
            .filter(|well| well.coords.len() == x.len())
            .map(|well| com_free_delta(x, well.coords.view()).1)
            .fold(f64::INFINITY, f64::min)
    })
}

/// \((E,g)\) seen by rgmin: identity when unarmed, \(E+V\) and
/// Householder-\(g+\nabla V\) when a Leave is in flight.
pub fn effective(x: ArrayView1<f64>, energy: f64, grad: Array1<f64>) -> (f64, Array1<f64>) {
    ARMED.with(|slot| {
        let mut held = slot.borrow_mut();
        let Some(armed) = held.as_mut() else {
            return (energy, grad);
        };
        armed.raw_energy = Some(energy);
        transform(armed, x, energy, grad)
    })
}

/// Raw \(E\) at the last point the armed transform saw.
fn raw_energy() -> Option<f64> {
    ARMED.with(|slot| slot.borrow().as_ref().and_then(|armed| armed.raw_energy))
}

fn transform(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    if armed.frozen && !armed.frozen_modes.is_empty() {
        return apply_packing_modes(armed, energy, grad);
    }
    if armed.wells.iter().any(|well| well.packing_mean.is_some()) {
        return transform_packing(armed, x, energy, grad);
    }
    transform_cartesian(armed, x, energy, grad)
}

/// Lift neighbour modes once, at arm. The hop applies this \(P\).
fn freeze_packing_modes(
    origin: ArrayView1<f64>,
    wells: &[Well],
) -> Vec<(usize, Array1<f64>, f64, f64)> {
    let Some(mu) = packing_mean(origin) else {
        return Vec::new();
    };
    let j = jacobian_nu3(origin, PACKING_SPEC, None);
    wells
        .iter()
        .enumerate()
        .filter_map(|(index, well)| {
            let mu_k = well.packing_mean.as_ref()?;
            let (p, r_phi, p_norm) = lift_packing_mode(origin, mu.view(), mu_k.view(), &j)?;
            Some((index, p, r_phi, p_norm))
        })
        .collect()
}

fn apply_packing_modes(
    armed: &mut Armed,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    if armed.lift.is_none() {
        let grain = crate::catalog::PACKING_LINK;
        let mut amplitude = 0.0;
        for (_, p, r_phi, p_norm) in &armed.frozen_modes {
            if *r_phi < 1e-12 || *p_norm < 1e-15 {
                continue;
            }
            let slope = dot(&grad, p) / p_norm;
            if slope > 0.0 {
                let curvature = slope / r_phi;
                let trial = 0.5 * curvature * grain * grain;
                if trial > amplitude {
                    amplitude = trial;
                }
            }
        }
        armed.lift = Some(amplitude);
        armed.sigma_phi = Some(grain);
    }
    let amplitude = armed.lift.unwrap_or(0.0);
    let sigma_phi = armed.sigma_phi.unwrap_or(0.0);
    let delta_t = armed.delta_t;
    let mut grad = if householder_suppressed() {
        grad
    } else {
        householder_packing(&armed.frozen_modes, sigma_phi, grad)
    };
    if sigma_phi <= 1e-12 {
        return (energy, grad);
    }
    let mut potential = 0.0;
    for (well, p, r_phi, p_norm) in &armed.frozen_modes {
        if *r_phi < 1e-12 {
            continue;
        }
        let entropy = armed.wells.get(*well).map_or(0.0, |held| held.entropy);
        let free = amplitude + entropy;
        if free <= 0.0 {
            continue;
        }
        let standing = armed.wells.get(*well).map_or(0.0, |held| held.deposit);
        let tempered = if delta_t > 1e-12 {
            free * (-standing / delta_t).exp()
        } else {
            free
        };
        let gauss = (-0.5 * (r_phi / sigma_phi) * (r_phi / sigma_phi)).exp();
        potential += tempered * gauss;
        let scale = -tempered * gauss * r_phi / (sigma_phi * sigma_phi) * p_norm;
        for (g, pk) in grad.iter_mut().zip(p.iter()) {
            *g += scale * *pk;
        }
    }
    (energy + potential, grad)
}

fn transform_packing(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    let Some(mu) = packing_mean(x) else {
        return (energy, grad);
    };
    let modes = packing_modes(armed, x, mu.view());
    if armed.lift.is_none() {
        // Depth of the well along the packing coordinate itself.
        //
        // \(r_\varphi=\|\mu-\mu_k\|\) is a descriptor distance and
        // \(\hat P\) is a unit Cartesian direction, so
        // \(\nabla E\cdot\hat P\) is a force and is not a slope in
        // \(r_\varphi\). The chain rule supplies the conversion:
        // \(\nabla_x r_\varphi=\|P\|\hat P\), so
        // \(\mathrm dE/\mathrm dr_\varphi=(\nabla E\cdot\hat P)/\|P\|\)
        // and a harmonic well of that slope at distance \(r_\varphi\)
        // is \(\tfrac12(\nabla E\cdot\hat P)r_\varphi/\|P\|\) deep.
        // Dropping \(\|P\|\) leaves a force times a descriptor length,
        // which is not an energy and cannot be compared with the barrier
        // the hill has to fill: on LJ75 that is 8.69 or 7.48
        // \(\varepsilon\) (Doye, Wales, Berry, *J. Chem. Phys.* **1995**,
        // *103*, 4234).
        // The length scale is the packing grain, not the current offset.
        //
        // Measuring the depth *at* \(r_\varphi\) prices the well at
        // whatever distance the walk happens to be standing, and the walk
        // arms one rung out: measured on LJ75 that is
        // \(r_\varphi=0.0109\), giving a hill of 0.029 \(\varepsilon\)
        // against a barrier of 8.69, and a width 63 times narrower than
        // the 0.69 that separates icosahedral from Marks. Neither number
        // can be fixed by running more chains.
        //
        // What the well has to be priced at is the distance where
        // packings stop being the same packing, which is
        // [`crate::catalog::PACKING_LINK`]. The harmonic form supplies
        // the extrapolation: the curvature is the measured slope over the
        // measured offset, \(\kappa=s/r_\varphi\), and the depth at the
        // grain is \(\tfrac12\kappa\sigma_g^2\). Reading it at the offset
        // instead is the same expression with \(\sigma_g\) replaced by
        // \(r_\varphi\), so it is short by \((\sigma_g/r_\varphi)^2\) --
        // a thousandfold here, and a different factor on every structure,
        // which is why it never looked like a constant to correct.
        let grain = crate::catalog::PACKING_LINK;
        let mut amplitude = 0.0;
        for (_, p, r_phi, p_norm) in &modes {
            if *r_phi < 1e-12 || *p_norm < 1e-15 {
                continue;
            }
            let slope = dot(&grad, p) / p_norm;
            if slope > 0.0 {
                let curvature = slope / r_phi;
                let trial = 0.5 * curvature * grain * grain;
                if trial > amplitude {
                    amplitude = trial;
                }
            }
        }
        armed.lift = Some(amplitude);
        armed.sigma_phi = Some(grain);
    }
    let amplitude = armed.lift.unwrap_or(0.0);
    let sigma_phi = armed.sigma_phi.unwrap_or(0.0);
    let delta_t = armed.delta_t;
    let mut grad = if householder_suppressed() {
        grad
    } else {
        householder_packing(&modes, sigma_phi, grad)
    };
    if sigma_phi <= 1e-12 {
        return (energy, grad);
    }
    let mut potential = 0.0;
    for (well, p, r_phi, p_norm) in &modes {
        if *r_phi < 1e-12 {
            continue;
        }
        // \(F_k=E_k-TS_k\). The depth `amplitude` is the potential half,
        // shared by every well because it is the curvature of the surface
        // the walk is on. The entropic half is per well: a packing the run
        // keeps arriving on costs \(T\ln n_k\) more to leave than its
        // depth alone says, and on LJ75 that term is what decides the
        // answer.
        let entropy = armed.wells.get(*well).map_or(0.0, |held| held.entropy);
        let free = amplitude + entropy;
        if free <= 0.0 {
            continue;
        }
        // Well-tempered scaling: a hill on a well that already carries
        // \(V_k\) is worth \(e^{-V_k/\Delta T}\) of a fresh one, so the
        // pile converges instead of growing without bound.
        let standing = armed.wells.get(*well).map_or(0.0, |held| held.deposit);
        let tempered = if delta_t > 1e-12 {
            free * (-standing / delta_t).exp()
        } else {
            free
        };
        let gauss = (-0.5 * (r_phi / sigma_phi) * (r_phi / sigma_phi)).exp();
        potential += tempered * gauss;
        // ∇_x r_φ = P_unnorm = ||P|| P̂. dV/dr_φ = −A gauss r_φ / σ².
        let scale = -tempered * gauss * r_phi / (sigma_phi * sigma_phi) * p_norm;
        for (g, pk) in grad.iter_mut().zip(p.iter()) {
            *g += scale * *pk;
        }
    }
    (energy + potential, grad)
}

/// Packing modes at `x`, one per well that carries a \(\nu=3\) mean.
///
/// \(\hat P\) and \(\|P\|\) come from the stacked Jacobian, which is
/// \((N d)\times 3N\) and is the whole cost of a transformed step, so they
/// are lifted again only once the geometry has moved by
/// [`crate::catalog::PACKING_MOVE_EPS`] -- the same staleness the packing
/// book uses on its histograms, and below the grain either can resolve.
/// \(r_\varphi\) is the distance in the map and is read fresh from
/// \(\mu\) on every call, so the hill and its gradient still track the
/// walk.
fn packing_modes(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    mu: ArrayView1<f64>,
) -> Vec<(usize, Array1<f64>, f64, f64)> {
    let stale = armed
        .mode_x
        .as_ref()
        .is_none_or(|held| moved(held.view(), x, crate::catalog::PACKING_MOVE_EPS));
    if stale {
        let j = jacobian_nu3(x, PACKING_SPEC, None);
        armed.modes = armed
            .wells
            .iter()
            .enumerate()
            .filter_map(|(index, well)| {
                let mu_k = well.packing_mean.as_ref()?;
                let (p, _, p_norm) = lift_packing_mode(x, mu, mu_k.view(), &j)?;
                Some((index, p, p_norm))
            })
            .collect();
        armed.mode_x = Some(x.to_owned());
    }
    armed
        .modes
        .iter()
        .filter_map(|(index, p, p_norm)| {
            let mu_k = armed.wells.get(*index)?.packing_mean.as_ref()?;
            if mu_k.len() != mu.len() {
                return None;
            }
            let r_phi = mu
                .iter()
                .zip(mu_k.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            Some((*index, p.clone(), r_phi, *p_norm))
        })
        .collect()
}

/// Whether any atom moved by more than `eps`.
fn moved(left: ArrayView1<f64>, right: ArrayView1<f64>, eps: f64) -> bool {
    if left.len() != right.len() {
        return true;
    }
    let limit = eps * eps;
    (0..left.len() / 3).any(|atom| {
        let d0 = left[3 * atom] - right[3 * atom];
        let d1 = left[3 * atom + 1] - right[3 * atom + 1];
        let d2 = left[3 * atom + 2] - right[3 * atom + 2];
        d0 * d0 + d1 * d1 + d2 * d2 > limit
    })
}

fn transform_cartesian(
    armed: &mut Armed,
    x: ArrayView1<f64>,
    energy: f64,
    grad: Array1<f64>,
) -> (f64, Array1<f64>) {
    let n_at = (x.len() / 3).max(1) as f64;
    let sigma = armed.sigma_rmsd;
    let mut potential = 0.0;
    if armed.lift.is_none() {
        // Amplitude that inverts the measured radial curvature of the
        // nearest known well: A = λ N σ² so H_V(well) = -λ I in the
        // RMSD metric. λ ≈ (g·û)/r at the Leave start.
        let mut amplitude: f64 = 0.0;
        for well in &armed.wells {
            if well.coords.len() != x.len() {
                continue;
            }
            let (delta, rmsd) = com_free_delta(x, well.coords.view());
            if rmsd < 1e-12 {
                continue;
            }
            let radius = n_at.sqrt() * rmsd;
            let uh_dot_g = dot(&grad, &delta) / radius;
            let lambda = uh_dot_g / radius;
            if lambda > 0.0 {
                amplitude = amplitude.max(lambda * n_at * sigma * sigma);
            }
        }
        armed.lift = Some(amplitude);
    }
    let amplitude = armed.lift.unwrap_or(0.0);
    let mut grad = if householder_suppressed() {
        grad
    } else {
        householder_cartesian(&armed.wells, x, grad)
    };
    if amplitude <= 0.0 {
        return (energy, grad);
    }
    for well in &armed.wells {
        if well.coords.len() != x.len() {
            continue;
        }
        let (delta, rmsd) = com_free_delta(x, well.coords.view());
        if rmsd < 1e-12 {
            continue;
        }
        let gauss = (-0.5 * (rmsd / sigma) * (rmsd / sigma)).exp();
        potential += amplitude * gauss;
        let scale = -amplitude * gauss / (n_at * sigma * sigma);
        for (g, d) in grad.iter_mut().zip(delta.iter()) {
            *g += scale * *d;
        }
    }
    (energy + potential, grad)
}

/// Invert the force component that would walk toward a known packing.
///
/// Henkelman–Jónsson on the pulled-back packing mode: \(g \leftarrow
/// g-2(g\cdot\hat P)\hat P\) when \(g\cdot\hat P>0\) (descent
/// \(-\nabla E\) points at \(\mu_k\)).
fn householder_packing(
    modes: &[(usize, Array1<f64>, f64, f64)],
    sigma_phi: f64,
    mut grad: Array1<f64>,
) -> Array1<f64> {
    // One reflection, along the direction that points at the known wells
    // together.
    //
    // Reflecting once per well, in place, is a product of reflections. The
    // pulled-back modes of different wells are not orthogonal, so reflecting
    // off the second restores part of the component toward the first, and
    // the result depends on the order the wells sit in the list: a product
    // of k reflections is an orthogonal transform of determinant
    // \((-1)^k\), which rotates the force rather than inverting its
    // approach. With one well on file that never showed, and one well is
    // what the cloud held until the chains began to interact.
    //
    // The wells are combined the way the hill combines them, by the same
    // Gaussian weight, so the reflection is dominated by the wells the
    // quench is actually near and reduces to Henkelman-Jonsson when one of
    // them dominates.
    let dim = grad.len();
    let mut aggregate = Array1::<f64>::zeros(dim);
    for (_, p, r_phi, _) in modes {
        if *r_phi < 1e-12 || p.len() != dim {
            continue;
        }
        let weight = if sigma_phi > 1e-12 {
            (-0.5 * (r_phi / sigma_phi) * (r_phi / sigma_phi)).exp()
        } else {
            1.0
        };
        aggregate.scaled_add(weight, p);
    }
    let norm = aggregate.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return grad;
    }
    aggregate /= norm;
    let proj = dot(&grad, &aggregate);
    if proj <= 0.0 {
        return grad;
    }
    let two = 2.0 * proj;
    for (g, pk) in grad.iter_mut().zip(aggregate.iter()) {
        *g -= two * *pk;
    }
    grad
}

/// Invert the force component that would walk toward a known well.
///
/// Cartesian fallback: \(g \leftarrow g-2(g\cdot\hat u)\hat u\) when
/// \(g\cdot\hat u>0\) (descent \(-\nabla E\) points at the well).
fn householder_cartesian(wells: &[Well], x: ArrayView1<f64>, mut grad: Array1<f64>) -> Array1<f64> {
    for well in wells {
        if well.coords.len() != x.len() {
            continue;
        }
        let (delta, rmsd) = com_free_delta(x, well.coords.view());
        if rmsd < 1e-12 {
            continue;
        }
        let n_at = (x.len() / 3).max(1) as f64;
        let u_norm = (n_at).sqrt() * rmsd;
        let proj = dot(&grad, &delta) / u_norm;
        if proj <= 0.0 {
            continue;
        }
        let two = 2.0 * proj / u_norm;
        for (g, d) in grad.iter_mut().zip(delta.iter()) {
            *g -= two * *d;
        }
    }
    grad
}

/// Mean of per-center SOAP+ACE \(\nu=3\) at [`PACKING_SPEC`].
fn packing_mean(x: ArrayView1<f64>) -> Option<Array1<f64>> {
    let n_at = x.len() / 3;
    if n_at < MINIMUM_PACKING_ATOMS {
        return None;
    }
    let loc = local_nu3_z(x, PACKING_SPEC, None);
    if loc.nrows() == 0 || loc.ncols() == 0 {
        return None;
    }
    let dim = loc.ncols();
    let mut mu = Array1::<f64>::zeros(dim);
    for i in 0..loc.nrows() {
        for t in 0..dim {
            mu[t] += loc[[i, t]];
        }
    }
    let n = loc.nrows() as f64;
    for item in mu.iter_mut() {
        *item /= n;
    }
    if mu.iter().copied().all(f64::is_finite) {
        Some(mu)
    } else {
        None
    }
}

/// Cartesian packing mode \(P=J_\mu^{\mathsf T}\hat u_\varphi\).
///
/// Returns \((\hat P, \|\mu-\mu_k\|, \|P\|)\) so the hill gradient
/// can reconstruct \(\nabla_x r_\varphi=\|P\|\hat P\).
fn packing_pullback(x: ArrayView1<f64>, mu_k: ArrayView1<f64>) -> Option<(Array1<f64>, f64, f64)> {
    let mu = packing_mean(x)?;
    let j = jacobian_nu3(x, PACKING_SPEC, None);
    lift_packing_mode(x, mu.view(), mu_k, &j)
}

fn lift_packing_mode(
    x: ArrayView1<f64>,
    mu: ArrayView1<f64>,
    mu_k: ArrayView1<f64>,
    j: &Array2<f64>,
) -> Option<(Array1<f64>, f64, f64)> {
    if mu.len() != mu_k.len() {
        return None;
    }
    let mut dmu = &mu - &mu_k;
    let r_phi = dmu.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r_phi < 1e-15 {
        return None;
    }
    for item in dmu.iter_mut() {
        *item /= r_phi;
    }
    let n_at = x.len() / 3;
    let dim = mu.len();
    if j.nrows() != n_at * dim || j.ncols() != x.len() {
        return None;
    }
    let mut p = Array1::<f64>::zeros(x.len());
    let inv_n = 1.0 / (n_at as f64);
    for k in 0..x.len() {
        let mut s = 0.0;
        for i in 0..n_at {
            for t in 0..dim {
                s += j[[i * dim + t, k]] * dmu[t];
            }
        }
        p[k] = s * inv_n;
    }
    strip_com(&mut p);
    let p_norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    if p_norm < 1e-15 {
        return None;
    }
    for item in p.iter_mut() {
        *item /= p_norm;
    }
    Some((p, r_phi, p_norm))
}

fn packing_l2(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn strip_com(delta: &mut Array1<f64>) {
    let n_at = delta.len() / 3;
    if n_at == 0 {
        return;
    }
    let mut com = [0.0; 3];
    for i in 0..n_at {
        for ax in 0..3 {
            com[ax] += delta[3 * i + ax];
        }
    }
    let n = n_at as f64;
    for item in &mut com {
        *item /= n;
    }
    for i in 0..n_at {
        for ax in 0..3 {
            delta[3 * i + ax] -= com[ax];
        }
    }
}

fn com_free_delta(x: ArrayView1<f64>, well: ArrayView1<f64>) -> (Array1<f64>, f64) {
    let n_at = x.len() / 3;
    let mut delta = Array1::zeros(x.len());
    let mut com = [0.0; 3];
    for i in 0..n_at {
        for ax in 0..3 {
            let d = x[3 * i + ax] - well[3 * i + ax];
            delta[3 * i + ax] = d;
            com[ax] += d;
        }
    }
    if n_at > 0 {
        for item in &mut com {
            *item /= n_at as f64;
        }
        for i in 0..n_at {
            for ax in 0..3 {
                delta[3 * i + ax] -= com[ax];
            }
        }
    }
    let n = n_at.max(1) as f64;
    let rmsd = (delta.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    (delta, rmsd)
}

fn same_point(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(a, b)| (a - b).abs() <= 1e-12)
}

fn dot(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

/// Fallback Leave size when no curvature is available, in Cartesian RMSD.
///
/// A length is the wrong thing to fix. At a minimum the gradient vanishes,
/// so a displacement of root-mean-square size \(\delta\) over \(N\) atoms
/// reaches \(\lambda N\delta^2/2\) under curvature \(\lambda\), and the
/// step that reaches a barrier \(\Delta\) is [`rung_rmsd`]. Measured by
/// Lanczos on the sealed LJ75 icosahedral minimum, \(\lambda_{\min}=12.83\),
/// so the 7.48 \(\varepsilon\) ico-Marks barrier of Wales and Doye is a
/// step of \(0.125\) and the energy it actually costs is \(7.11\). This
/// constant is nearly three times that step, and reach grows as
/// \(\delta^2\), so along the same mode it spends 58.9 \(\varepsilon\):
/// eight times the barrier, which is a melt and not a crossing. It survives
/// only for callers that cannot afford a curvature pass.
///
/// Wales, D. J.; Doye, J. P. K. *J. Phys. Chem. A* **1997**, *101*, 5111.
/// <https://doi.org/10.1021/jp970984n>
pub const LEAVE_RUNG_RMSD: f64 = 0.35;

/// Barrier the first rung aims at, in units of the well depth per atom.
///
/// The ladder is walked in energy, not in length, because a barrier is what
/// a Leave has to clear and the step that clears it depends on the curvature
/// of the structure it starts from. The unit is the depth per atom of the
/// minimum being left, which every cluster potential supplies and into which
/// no morphology enters.
pub const LEAVE_BARRIER_FLOOR: f64 = 0.25;

/// Ratio between the barriers successive rungs aim at.
pub const LEAVE_BARRIER_GROWTH: f64 = 2.0;

/// Root-mean-square step whose harmonic reach is `barrier`.
///
/// \(\delta=\sqrt{2\Delta/(\lambda N)}\), which is
/// `Hop.rung_reaches_barrier` in `proofs/lean/Hop/LeavePacking.lean`. `None`
/// when the curvature, the count or the barrier is not positive, so a caller
/// with no measurement falls back rather than inventing a length.
pub fn rung_rmsd(curvature: f64, atoms: usize, barrier: f64) -> Option<f64> {
    if !(curvature > 0.0) || atoms == 0 || !(barrier > 0.0) {
        return None;
    }
    let delta = (2.0 * barrier / (curvature * atoms as f64)).sqrt();
    delta.is_finite().then_some(delta)
}

/// Barrier the rung at index `rung` aims at, from the well depth per atom.
pub fn rung_barrier(depth_per_atom: f64, rung: usize) -> f64 {
    depth_per_atom.abs() * LEAVE_BARRIER_FLOOR * LEAVE_BARRIER_GROWTH.powi(rung as i32)
}

/// Rungs a single Leave walks before it reports a refusal.
pub const LEAVE_RUNGS: usize = 6;

/// Independent covering starts launched in the packing active volume
/// on one Leave.
///
/// Xu, Osetsky and Stoller (*Phys. Rev. B* **84**, 132103 (2011))
/// run several dimer searches per active volume. Béland, Osetsky,
/// Stoller and Xu (arXiv:1409.1253): SEAKMC samples tens of events
/// then flushes the catalog after one is executed. Occupancy is not
/// kMC and does not keep the unused starts.
pub const LEAVE_AV_SEARCHES: usize = 8;

/// Rungs walked past the first escape before the best of them is taken.
///
/// The first rung that leaves is not the best one to leave by. Measured from
/// the LJ75 icosahedral minimum, the first escape lands anywhere between 5
/// and 25 \(\varepsilon\) above the floor, and the spread across
/// neighbouring rungs of one direction is most of that range.
pub const LEAVE_RUNG_EXTRA: usize = 2;

/// Unit packing-map direction: covering point `cover_index`, orthogonal
/// to \(\mu\), signed away from the nearest packing on file.
///
/// Plasencia Gutiérrez, M.; Argáez, C.; Jónsson, H. *J. Chem. Theory
/// Comput.* **2017**, *13* (1), 125-134.
/// <https://doi.org/10.1021/acs.jctc.5b01216>
pub fn packing_cover_direction(
    x: ArrayView1<f64>,
    cover_index: usize,
    references: &[Vec<f64>],
) -> Option<Vec<f64>> {
    let mu = packing_mean(x)?;
    let dim = mu.len();
    if dim == 0 {
        return None;
    }
    let n_cover = crate::hypersphere::default_cover_size();
    let mut direction = crate::hypersphere::cover_direction(n_cover, dim, cover_index);
    if direction.len() != dim {
        return None;
    }
    orthonormalize_against_mean(&mut direction, mu.view())?;
    // Away from the nearest packing on file: the sign of a covering point is
    // arbitrary, and half of them point back at a well the run already holds.
    let mut nearest: Option<(f64, Array1<f64>)> = None;
    for reference in references {
        if reference.len() != x.len() {
            continue;
        }
        let Some(mu_k) = packing_mean(ArrayView1::from(reference.as_slice())) else {
            continue;
        };
        if mu_k.len() != dim {
            continue;
        }
        let distance = mu
            .iter()
            .zip(mu_k.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        if nearest.as_ref().is_none_or(|(held, _)| distance < *held) {
            nearest = Some((distance, mu_k));
        }
    }
    if let Some((distance, mu_k)) = nearest
        && distance > 1e-12
    {
        let away = direction
            .iter()
            .zip(mu.iter().zip(mu_k.iter()))
            .map(|(d, (a, b))| d * (a - b))
            .sum::<f64>();
        if away < 0.0 {
            for value in &mut direction {
                *value = -*value;
            }
        }
    }
    Some(direction)
}

/// Unit packing-map direction from `from` toward `toward`.
///
/// \(\widehat{\mu_{\mathrm{to}}-\mu_{\mathrm{from}}}\) with the breath
/// along \(\mu_{\mathrm{from}}\) removed. This is the increment that
/// walks one known packing onto another in the same map the covering
/// Leave uses; a covering of \(S^{d-1}\) at \(d\sim 10^{2}\) does not
/// hit it by chance.
pub fn packing_direction_between(
    from: ArrayView1<f64>,
    toward: ArrayView1<f64>,
) -> Option<Vec<f64>> {
    let mu = packing_mean(from)?;
    let mu_t = packing_mean(toward)?;
    if mu.len() != mu_t.len() || mu.is_empty() {
        return None;
    }
    let mut direction: Vec<f64> = mu.iter().zip(mu_t.iter()).map(|(a, b)| b - a).collect();
    orthonormalize_against_mean(&mut direction, mu.view())?;
    Some(direction)
}

fn orthonormalize_against_mean(direction: &mut [f64], mu: ArrayView1<f64>) -> Option<()> {
    if direction.len() != mu.len() {
        return None;
    }
    let mu_norm2 = mu.iter().map(|v| v * v).sum::<f64>();
    if mu_norm2 > 1e-15 {
        let projection = direction
            .iter()
            .zip(mu.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        for (value, mean) in direction.iter_mut().zip(mu.iter()) {
            *value -= projection * *mean / mu_norm2;
        }
    }
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return None;
    }
    for value in direction.iter_mut() {
        *value /= norm;
    }
    Some(())
}

/// One rung of the Leave ladder: a packing-map step of Cartesian size
/// `rmsd` along the covering direction `cover_index`, pointed away from the
/// packings on file.
pub fn leave_packing_rung(
    x: ArrayView1<f64>,
    cover_index: usize,
    rmsd: f64,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Array1<f64> {
    let Some(direction) = packing_cover_direction(x, cover_index, references) else {
        return x.to_owned();
    };
    crate::soap::packing_step_nu3(x, PACKING_SPEC, &direction, rmsd, species, mobile)
}

/// Curvature of the potential along a unit Cartesian direction.
///
/// \(\lambda_u = \hat u^{\mathsf T} H \hat u\) by central difference of
/// the gradient, two evaluations. Reported for the record; the rung is sized
/// by [`leave_packing_rung_to`], which measures the energy instead of
/// trusting this.
pub fn direction_curvature<G>(
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    epsilon: f64,
    mut grad: G,
) -> Option<f64>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm > 1e-15) || !(epsilon > 0.0) {
        return None;
    }
    let mut plus = x.to_owned();
    let mut minus = x.to_owned();
    for ((p, m), d) in plus.iter_mut().zip(minus.iter_mut()).zip(direction.iter()) {
        let step = epsilon * *d / norm;
        *p += step;
        *m -= step;
    }
    let gp = grad(plus.view())?;
    let gm = grad(minus.view())?;
    if gp.len() != direction.len() || gm.len() != direction.len() {
        return None;
    }
    let curvature = gp
        .iter()
        .zip(gm.iter())
        .zip(direction.iter())
        .map(|((a, b), d)| (a - b) * *d / norm)
        .sum::<f64>()
        / (2.0 * epsilon);
    curvature.is_finite().then_some(curvature)
}

/// Bracketing and bisection steps the rung line search may take.
pub const LEAVE_RUNG_BACKTRACKS: usize = 12;

/// Rung sized so the measured energy rise is at most `barrier`.
///
/// A root-mean-square cap is the wrong bound for this direction. The packing
/// pullback is not spread over the cluster: it concentrates on the few
/// centres whose environment the increment changes, so an RMSD of 0.35 over
/// 75 atoms can be one atom moving 1.7 sigma, straight through its
/// neighbours. A crushed cluster has enormous *negative* transverse
/// curvature, \(-r^{-1}\,\mathrm{d}V/\mathrm{d}r\) with the pair force
/// deep in the repulsive wall, so a min-mode climb started there reports
/// \(\lambda<0\) and a flipped force on its first step and calls the ridge
/// crossed. Measured on the sealed LJ75 icosahedral minimum: every one of
/// twelve covering starts declared a crossing after one step at curvatures
/// between \(-7\times10^5\) and \(-10^{13}\), and the quench from there
/// returned to the floor it started on.
///
/// [`rung_rmsd`] supplies the first trial length from the harmonic identity,
/// and the length is then halved until the potential itself agrees that the
/// step costs no more than the barrier. That bound cannot be met by a
/// crushed structure, so the climb always starts on the landscape.
pub fn leave_packing_rung_to<E>(
    x: ArrayView1<f64>,
    cover_index: usize,
    barrier: f64,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    energy: E,
) -> Array1<f64>
where
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let Some(direction) = packing_cover_direction(x, cover_index, references) else {
        return x.to_owned();
    };
    leave_packing_rung_to_dir(x, &direction, barrier, species, mobile, energy)
}

/// Cover index whose packing-mean increment is farthest from every
/// occupied packing mean.
///
/// Anelli, Engel, Pickard and Ceriotti (*Phys. Rev. Materials* **2**,
/// 103804 (2018), <https://doi.org/10.1103/PhysRevMaterials.2.103804>):
/// farthest-point sampling in a learned descriptor, not a random
/// cover. Occupancy already has [`crate::catalog_policy::proposal::farthest_hole`]
/// on the unit sphere; this is the same rule on the packing-mean
/// sphere the Leave actually walks.
pub fn farthest_packing_cover(
    x: ArrayView1<f64>,
    references: &[Vec<f64>],
    n_cover: usize,
) -> usize {
    let Some(mu) = packing_mean(x) else {
        return 0;
    };
    let occupied: Vec<Array1<f64>> = references
        .iter()
        .filter_map(|reference| packing_mean(ArrayView1::from(reference.as_slice())))
        .filter(|mean| mean.len() == mu.len())
        .collect();
    let mut best_i = 0usize;
    let mut best_d = f64::NEG_INFINITY;
    for index in 0..n_cover.max(1) {
        let Some(direction) = packing_cover_direction(x, index, references) else {
            continue;
        };
        if direction.len() != mu.len() {
            continue;
        }
        let dmin = if occupied.is_empty() {
            direction.iter().map(|v| v * v).sum::<f64>().sqrt()
        } else {
            occupied
                .iter()
                .map(|occ| {
                    direction
                        .iter()
                        .zip(mu.iter().zip(occ.iter()))
                        .map(|(d, (m, o))| {
                            let t = m + d - o;
                            t * t
                        })
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(f64::INFINITY, f64::min)
        };
        if dmin > best_d {
            best_d = dmin;
            best_i = index;
        }
    }
    best_i
}

/// Energy minima on a sphere supported on the active volume.
///
/// Ohno and Maeda (*Chem. Phys. Lett.* **384**, 277 (2004),
/// <https://doi.org/10.1016/j.cplett.2003.12.030>): reaction
/// channels are anharmonic downward distortions, found as energy
/// minima on the scaled hypersphere around the equilibrium. A
/// uniform covering is not that. This scan evaluates the potential
/// on a sphere of radius `radius` in the mobile coordinates and
/// returns the `keep` lowest-energy points. Later SEAKMC builds
/// used the same SHS start (Xu *et al.*, *Comput. Mater. Sci.*
/// **194**, 110390 (2021)).
pub fn shs_av_starts<E>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    radius: f64,
    samples: usize,
    keep: usize,
    mut energy: E,
) -> Vec<Array1<f64>>
where
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let n_at = x.len() / 3;
    if n_at == 0 || mobile.is_empty() || !(radius.is_finite() && radius > 0.0) || samples == 0 {
        return Vec::new();
    }
    let dim = 3 * mobile.len();
    let mut scored: Vec<(f64, Array1<f64>)> = Vec::with_capacity(samples);
    for index in 0..samples {
        let direction = crate::hypersphere::cover_direction(samples, dim, index);
        if direction.len() != dim {
            continue;
        }
        let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            continue;
        }
        let scale = radius / norm;
        let mut trial = x.to_owned();
        for (slot, &atom) in mobile.iter().enumerate() {
            if atom >= n_at {
                continue;
            }
            for k in 0..3 {
                trial[3 * atom + k] += scale * direction[3 * slot + k];
            }
        }
        let Some(e) = energy(trial.view()) else {
            continue;
        };
        if e.is_finite() {
            scored.push((e, trial));
        }
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(keep.max(1))
        .map(|(_, state)| state)
        .collect()
}

/// Inverse-power in Maeda's pair weight \(\omega_{ij}=[(R_i+R_j)/r_{ij}]^p\).
pub const AFIR_P: i32 = 6;

/// Collision radius when no species table applies: one Lennard-Jones
/// \(\sigma\). Occupancy Leave is run in reduced units.
pub const AFIR_DEFAULT_RADIUS: f64 = 1.0;

/// Pair radii for the AFIR weight. Occupancy Leave is in reduced units,
/// so every centre is one \(\sigma\) whether or not a species list is
/// supplied.
pub fn afir_radii(n_at: usize, _species: Option<&[u32]>) -> Vec<f64> {
    vec![AFIR_DEFAULT_RADIUS; n_at]
}

/// Maeda AFIR term and Cartesian gradient.
///
/// Maeda, Taketsugu and Morokuma (*J. Comput. Chem.* **35**, 166 (2014),
/// <https://doi.org/10.1002/jcc.23481>):
/// \(F_{\mathrm{AFIR}}=E+\rho\alpha\sum_{i\in A}\sum_{j\in B}\omega_{ij}r_{ij}/\sum\omega_{ij}\)
/// with \(\omega_{ij}=[(R_i+R_j)/r_{ij}]^6\). \(\rho=+1\) pushes the
/// fragments together; \(\rho=-1\) peels them. Occupancy takes
/// \(A\) as the packing active volume and \(B\) as its complement, so
/// the fragments are leftover versus core, not a named morphology.
pub fn afir_term(
    x: ArrayView1<f64>,
    fragment_a: &[usize],
    fragment_b: &[usize],
    rho: f64,
    alpha: f64,
    radii: &[f64],
) -> Option<(f64, Array1<f64>)> {
    let n_at = x.len() / 3;
    if n_at == 0
        || fragment_a.is_empty()
        || fragment_b.is_empty()
        || radii.len() < n_at
        || !rho.is_finite()
        || !alpha.is_finite()
    {
        return None;
    }
    let p = f64::from(AFIR_P);
    let mut num = 0.0;
    let mut den = 0.0;
    let mut pairs: Vec<(usize, usize, f64, f64, [f64; 3])> = Vec::new();
    for &i in fragment_a {
        if i >= n_at {
            continue;
        }
        for &j in fragment_b {
            if j >= n_at || i == j {
                continue;
            }
            let mut u = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                u[k] = d;
                r2 += d * d;
            }
            if !(r2.is_finite() && r2 > 1e-16) {
                continue;
            }
            let r = r2.sqrt();
            let rij = radii[i] + radii[j];
            if !(rij.is_finite() && rij > 0.0) {
                continue;
            }
            let ratio = rij / r;
            let omega = ratio.powi(AFIR_P);
            if !omega.is_finite() {
                continue;
            }
            num += omega * r;
            den += omega;
            pairs.push((i, j, r, omega, u));
        }
    }
    if !(den.is_finite() && den > 0.0 && num.is_finite()) {
        return None;
    }
    let mean = num / den;
    let value = rho * alpha * mean;
    if !value.is_finite() {
        return None;
    }
    let mut grad = Array1::zeros(x.len());
    let scale = rho * alpha / (den * den);
    for (i, j, r, omega, u) in pairs {
        // ω = (R/r)^p, ωr = R^p r^{1-p}
        // d(ωr)/dr = (1-p) ω, dω/dr = -p ω / r
        let dnum = (1.0 - p) * omega;
        let dden = -p * omega / r;
        let dmean = scale * (dnum * den - num * dden);
        if !dmean.is_finite() {
            continue;
        }
        for k in 0..3 {
            let component = dmean * u[k] / r;
            grad[3 * i + k] += component;
            grad[3 * j + k] -= component;
        }
    }
    if grad.iter().any(|g: &f64| !g.is_finite()) {
        return None;
    }
    Some((value, grad))
}

/// \(\alpha\) that puts \(|V_{\mathrm{AFIR}}|\) on `barrier` at `x`.
///
/// Maeda sizes \(\alpha\) from an Ar–Ar collision energy. Occupancy
/// works in reduced units and sizes \(\alpha\) so the artificial term
/// at the live well equals one Leave rung, which keeps the start on
/// the landscape instead of crushing the pair wall.
pub fn afir_alpha_for_barrier(
    x: ArrayView1<f64>,
    fragment_a: &[usize],
    fragment_b: &[usize],
    barrier: f64,
    radii: &[f64],
) -> Option<f64> {
    if !(barrier.is_finite() && barrier > 0.0) {
        return None;
    }
    let (mean, _) = afir_term(x, fragment_a, fragment_b, 1.0, 1.0, radii)?;
    let width = mean.abs();
    if !(width.is_finite() && width > 1e-12) {
        return None;
    }
    Some(barrier / width)
}

/// SC-AFIR starts on the packing active volume.
///
/// Minimize \(E+\rho\alpha\langle r\rangle_{A,B}\) from the live well
/// for \(\rho=\pm 1\), then drop the force. The two geometries are
/// starts; the real PES decides the path (Maeda *et al.*,
/// <https://doi.org/10.1002/jcc.23481>; GRRM17,
/// <https://doi.org/10.1002/jcc.25106>). \(A\) is `mobile`, \(B\) is
/// the complement.
pub fn afir_av_starts<F>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    barrier: f64,
    steps: usize,
    species: Option<&[u32]>,
    mut energy_grad: F,
) -> Vec<Array1<f64>>
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let n_at = x.len() / 3;
    if n_at == 0 || mobile.is_empty() || steps == 0 {
        return Vec::new();
    }
    let mut in_a = vec![false; n_at];
    for &i in mobile {
        if i < n_at {
            in_a[i] = true;
        }
    }
    let fragment_a: Vec<usize> = (0..n_at).filter(|&i| in_a[i]).collect();
    let fragment_b: Vec<usize> = (0..n_at).filter(|&i| !in_a[i]).collect();
    if fragment_a.is_empty() || fragment_b.is_empty() {
        return Vec::new();
    }
    let radii = afir_radii(n_at, species);
    let Some(alpha) = afir_alpha_for_barrier(x, &fragment_a, &fragment_b, barrier, &radii) else {
        return Vec::new();
    };
    let mut starts = Vec::with_capacity(2);
    for rho in [1.0_f64, -1.0] {
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (_, trial, _) = opt.minimize(x, steps, |trial| {
            let (energy, gradient) = energy_grad(trial)?;
            let (bias, bias_g) = afir_term(trial, &fragment_a, &fragment_b, rho, alpha, &radii)?;
            let total_e = energy + bias;
            let total_g = gradient + bias_g;
            if total_e.is_finite() && total_g.iter().all(|g: &f64| g.is_finite()) {
                Some((total_e, total_g))
            } else {
                None
            }
        });
        if trial.len() == x.len() && trial.iter().all(|v: &f64| v.is_finite()) {
            starts.push(trial);
        }
    }
    starts
}

/// Neighbour cutoff the hop uses in reduced Lennard-Jones units.
pub const LEAVE_NEIGHBOUR_CUTOFF: f64 = 1.6;

fn coordination(x: ArrayView1<f64>, cutoff: f64) -> Vec<usize> {
    let n = x.len() / 3;
    let cut2 = cutoff * cutoff;
    let mut coord = vec![0usize; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = x[3 * a + k] - x[3 * b + k];
                d2 += d * d;
            }
            if d2 < cut2 {
                coord[a] += 1;
                coord[b] += 1;
            }
        }
    }
    coord
}

fn worst_mobile(mobile: &[usize], coord: &[usize]) -> Option<usize> {
    mobile
        .iter()
        .copied()
        .filter(|&a| a < coord.len())
        .min_by_key(|&a| coord[a])
}

fn unit3<R: rand::Rng + ?Sized>(rng: &mut R) -> [f64; 3] {
    loop {
        let x = rng.random::<f64>() * 2.0 - 1.0;
        let y = rng.random::<f64>() * 2.0 - 1.0;
        let z = rng.random::<f64>() * 2.0 - 1.0;
        let n2 = x * x + y * y + z * z;
        if n2 > 1e-12 {
            let n = n2.sqrt();
            return [x / n, y / n, z / n];
        }
    }
}

/// Least-coordinated leftover atom placed on the outer shell.
///
/// This is [`crate::movekernel::SurfaceRelocate`] with the mover taken
/// from the packing active volume, not from the whole cluster.
pub fn leave_av_surface<R: rand::Rng + ?Sized>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    neighbour_cutoff: f64,
    rng: &mut R,
) -> Array1<f64> {
    let n = x.len() / 3;
    let Some(mover) = worst_mobile(mobile, &coordination(x, neighbour_cutoff)) else {
        return x.to_owned();
    };
    if n < 2 || mover >= n {
        return x.to_owned();
    }
    let mut c = [0.0; 3];
    for a in 0..n {
        for k in 0..3 {
            c[k] += x[3 * a + k];
        }
    }
    let inv = 1.0 / n as f64;
    for k in 0..3 {
        c[k] *= inv;
    }
    let mut shell: f64 = 0.0;
    for a in 0..n {
        if a == mover {
            continue;
        }
        let mut d2 = 0.0;
        for k in 0..3 {
            let d = x[3 * a + k] - c[k];
            d2 += d * d;
        }
        shell = shell.max(d2.sqrt());
    }
    let dir = unit3(rng);
    let r = shell * (0.85 + 0.20 * rng.random::<f64>());
    let mut out = x.to_owned();
    for k in 0..3 {
        out[3 * mover + k] = c[k] + dir[k] * r;
    }
    out
}

/// Least-coordinated leftover atom moved onto the best hollow site.
///
/// Shao, Cheng and Cai (*J. Comput. Chem.* **25**, 1693 (2004),
/// <https://doi.org/10.1002/jcc.20096>): the lattice is read off the
/// live structure. Occupancy only restricts which atom may move.
pub fn leave_av_hollow<R: rand::Rng + ?Sized>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    neighbour_cutoff: f64,
    rng: &mut R,
) -> Array1<f64> {
    let n = x.len() / 3;
    let Some(mover) = worst_mobile(mobile, &coordination(x, neighbour_cutoff)) else {
        return x.to_owned();
    };
    let kernel = crate::movekernel::HollowRelocate {
        n_points: n,
        neighbour_cutoff,
    };
    let sites = kernel.sites(x, mover);
    let Some(best) = sites.iter().map(|(_, c)| *c).max() else {
        return leave_av_surface(x, mobile, neighbour_cutoff, rng);
    };
    let candidates: Vec<&([f64; 3], usize)> = sites.iter().filter(|(_, c)| *c == best).collect();
    let pick = candidates[rng.random_range(0..candidates.len())].0;
    let mut out = x.to_owned();
    for k in 0..3 {
        out[3 * mover + k] = pick[k];
    }
    out
}

/// Repeated hollow moves of leftover atoms until the surface saturates.
pub fn leave_av_fill<R: rand::Rng + ?Sized>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    neighbour_cutoff: f64,
    max_moves: usize,
    rng: &mut R,
) -> Array1<f64> {
    let n = x.len() / 3;
    if n < 4 || mobile.is_empty() {
        return leave_av_hollow(x, mobile, neighbour_cutoff, rng);
    }
    let kernel = crate::movekernel::HollowRelocate {
        n_points: n,
        neighbour_cutoff,
    };
    let mut cur = x.to_owned();
    let mut moved = false;
    for _ in 0..max_moves.max(1) {
        let coord = coordination(cur.view(), neighbour_cutoff);
        let Some(mover) = worst_mobile(mobile, &coord) else {
            break;
        };
        let sites = kernel.sites(cur.view(), mover);
        let Some(best) = sites.iter().map(|(_, c)| *c).max() else {
            break;
        };
        if best <= coord[mover] {
            break;
        }
        let candidates: Vec<&([f64; 3], usize)> =
            sites.iter().filter(|(_, c)| *c == best).collect();
        let pick = candidates[rng.random_range(0..candidates.len())].0;
        for k in 0..3 {
            cur[3 * mover + k] = pick[k];
        }
        moved = true;
    }
    if moved {
        cur
    } else {
        leave_av_hollow(x, mobile, neighbour_cutoff, rng)
    }
}

/// One packing-changing step on the leftover. `kind` cycles hollow,
/// fill, surface, shell.
pub fn leave_av_step<R: rand::Rng + ?Sized>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    neighbour_cutoff: f64,
    kind: usize,
    rng: &mut R,
) -> Array1<f64> {
    let n = x.len() / 3;
    match kind % 4 {
        0 => leave_av_hollow(x, mobile, neighbour_cutoff, rng),
        1 => leave_av_fill(x, mobile, neighbour_cutoff, 12, rng),
        2 => leave_av_surface(x, mobile, neighbour_cutoff, rng),
        _ => crate::movekernel::ShellRotate { n_points: n }.propose(x, 0.0, rng),
    }
}

/// Packing-changing Leave starts on the leftover: hollow, fill, surface,
/// then a shell twist. Sphere covers and AFIR stay available; they do
/// not leave this funnel under a raw quench.
pub fn leave_av_packing_starts<R: rand::Rng + ?Sized>(
    x: ArrayView1<f64>,
    mobile: &[usize],
    neighbour_cutoff: f64,
    count: usize,
    rng: &mut R,
) -> Vec<Array1<f64>> {
    (0..count)
        .map(|k| leave_av_step(x, mobile, neighbour_cutoff, k, rng))
        .collect()
}

/// Hops one Leave walks, adopting ico-isomers so the next move is not
/// from the same geometry. Serial finds Marks on this library after
/// thousands of accepted hops; eight independent starts from the floor
/// do not.
pub const LEAVE_WALK_HOPS: usize = 32;

/// Hop temperature used for the walk accept. Same number as the LJ
/// cluster search preset.
pub const LEAVE_WALK_TEMPERATURE: f64 = 0.8;

/// Walk packing-changing hops from `origin`. Each hop rebuilds the
/// leftover on the *current* landing, proposes, quenches, and Metropolis-
/// accepts at [`LEAVE_WALK_TEMPERATURE`]. The return is the lowest-energy
/// quench that leaves the origin packing and is at or below the origin
/// energy. High-energy packings are walked through, not installed.
pub fn leave_av_walk<Q, R>(
    origin: ArrayView1<f64>,
    neighbour_cutoff: f64,
    hops: usize,
    origin_energy: f64,
    rng: &mut R,
    mut quench: Q,
) -> Option<(f64, Array1<f64>, usize)>
where
    Q: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    R: rand::Rng + ?Sized,
{
    let Some(origin_slice) = origin.as_slice() else {
        return None;
    };
    let mut current = origin.to_owned();
    let mut current_energy = origin_energy;
    let mut best: Option<(f64, Array1<f64>, usize)> = None;
    for hop in 0..hops.max(1) {
        let mobile = crate::soap::packing_active_volume(current.view(), PACKING_SPEC, None);
        let start = leave_av_step(current.view(), &mobile, neighbour_cutoff, hop, rng);
        let (energy, landed) = quench(start.view());
        if !energy.is_finite() || landed.len() != origin.len() {
            continue;
        }
        let left = landed
            .as_slice()
            .is_some_and(|trial| crate::catalog::leaves_packing(origin_slice, trial, &[]));
        if left && energy <= origin_energy + 1e-6 {
            if best.as_ref().is_none_or(|(held, _, _)| energy < *held) {
                best = Some((energy, landed.clone(), hop));
            }
        }
        let delta = energy - current_energy;
        let take = delta <= 0.0 || rng.random::<f64>() < (-delta / LEAVE_WALK_TEMPERATURE).exp();
        if take {
            current = landed;
            current_energy = energy;
        }
    }
    best
}

/// [`leave_packing_rung_to`] along a packed feature direction already in hand.
pub fn leave_packing_rung_to_dir<E>(
    x: ArrayView1<f64>,
    direction: &[f64],
    barrier: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    mut energy: E,
) -> Array1<f64>
where
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let Some(base) = energy(x).filter(|value| value.is_finite()) else {
        return x.to_owned();
    };
    let atoms = x.len() / 3;
    let mut rise = |rmsd: f64, energy: &mut E| -> Option<(f64, Array1<f64>)> {
        let trial =
            crate::soap::packing_step_nu3(x, PACKING_SPEC, direction, rmsd, species, mobile);
        let value = energy(trial.view())?;
        value.is_finite().then_some((value - base, trial))
    };
    // The step wanted is the *largest* one whose rise is still under the
    // barrier. Taking the first length that fits undershoots by whatever the
    // bracketing stride was: measured from the LJ75 icosahedral minimum, a
    // rung aiming at 42.3 eps delivered 12.7 at RMSD 0.066, four halvings
    // below where it belonged, and every quench returned to the floor.
    // Bracket first, then bisect.
    let guess = rung_rmsd(1.0, atoms, barrier).unwrap_or(LEAVE_RUNG_RMSD);
    // Bracket so that `lo` is the largest length known to fit under the
    // barrier and `hi` the smallest known to exceed it, then bisect. The two
    // bounds have to stay distinct: collapsing them onto the same value
    // leaves the step wherever the bracketing stride happened to stop, which
    // is what left a rung aiming at 42.3 eps spending 12.7.
    let mut lo = 0.0_f64;
    let mut hi = guess;
    let mut best: Option<Array1<f64>> = None;
    match rise(hi, &mut energy) {
        Some((value, trial)) if value <= barrier => {
            best = Some(trial);
            lo = hi;
            for _ in 0..LEAVE_RUNG_BACKTRACKS {
                hi = lo * 2.0;
                match rise(hi, &mut energy) {
                    Some((value, trial)) if value <= barrier => {
                        best = Some(trial);
                        lo = hi;
                    }
                    _ => break,
                }
            }
        }
        _ => {
            let mut probe = hi;
            for _ in 0..LEAVE_RUNG_BACKTRACKS {
                probe *= 0.5;
                match rise(probe, &mut energy) {
                    Some((value, trial)) if value <= barrier => {
                        best = Some(trial);
                        lo = probe;
                        break;
                    }
                    _ => hi = probe,
                }
            }
        }
    }
    if best.is_none() || !(hi > lo) {
        return best.unwrap_or_else(|| x.to_owned());
    }
    for _ in 0..LEAVE_RUNG_BACKTRACKS {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo) || !(mid < hi) {
            break;
        }
        match rise(mid, &mut energy) {
            Some((value, trial)) if value <= barrier => {
                best = Some(trial);
                lo = mid;
            }
            _ => hi = mid,
        }
    }
    best.unwrap_or_else(|| x.to_owned())
}

/// Packing-map steps one Leave may accumulate along one direction.
///
/// The ceiling at [`LEAVE_WALK_CLIMB`] times the depth per atom is what
/// ends the walk. This count lets that ceiling fire first at the first
/// rung's barrier.
pub const LEAVE_RIDGE_RUNGS: usize = 24;

/// Follow one packing-map covering direction in barrier-sized steps.
///
/// Each step is [`leave_packing_rung_to`] from the *current* point, not
/// from the well. The accept quench is raw \(E\): an invert-armed
/// landing is not a packing. Quench starts only after the unquenched
/// rise has reached [`rung_barrier`] at rung 2, which on LJ75 is past
/// the Wales–Doye ico–Marks barrier scale. Measured covering ladders
/// from the sealed icosahedral minimum leave to novel packings at
/// \(-371\) to \(-391\); this walk is the same increment, accumulated
/// far enough that a quench can sit on the far side.
pub fn leave_packing_ridge<R>(
    origin: ArrayView1<f64>,
    cover_index: usize,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    depth_per_atom: f64,
    relax_steps: usize,
    eval: R,
) -> Option<(f64, Array1<f64>, usize)>
where
    R: FnMut(ArrayView1<f64>, usize) -> (f64, Array1<f64>),
{
    leave_packing_walk(
        origin,
        references,
        species,
        mobile,
        depth_per_atom,
        relax_steps,
        eval,
        |here, energy| {
            leave_packing_rung_to(
                here,
                cover_index,
                rung_barrier(depth_per_atom, 0),
                references,
                species,
                mobile,
                energy,
            )
        },
    )
}

/// Walk the packing map from `origin` toward `target`'s packing mean.
///
/// Same energy-capped steps as [`leave_packing_ridge`], but the
/// direction is \(\widehat{\mu_{\mathrm{target}}-\mu}\) recomputed at
/// the current point rather than a covering index. A covering of the
/// high-dimensional packing sphere does not hit this vector.
pub fn leave_packing_toward<R>(
    origin: ArrayView1<f64>,
    target: ArrayView1<f64>,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    depth_per_atom: f64,
    relax_steps: usize,
    eval: R,
) -> Option<(f64, Array1<f64>, usize)>
where
    R: FnMut(ArrayView1<f64>, usize) -> (f64, Array1<f64>),
{
    leave_packing_walk(
        origin,
        references,
        species,
        mobile,
        depth_per_atom,
        relax_steps,
        eval,
        |here, energy| {
            let Some(direction) = packing_direction_between(here, target) else {
                return here.to_owned();
            };
            leave_packing_rung_to_dir(
                here,
                &direction,
                rung_barrier(depth_per_atom, 0),
                species,
                mobile,
                energy,
            )
        },
    )
}

/// Unquenched Leave starts scored by packing histogram, for FunnelModel EI.
///
/// Each probe is one first-rung step or the fivefold residual. The
/// quench is paid only for the cover EI (or Thompson) then selects.
pub fn propose_leave_covers<E, R>(
    origin: ArrayView1<f64>,
    references: &[Vec<f64>],
    depth_per_atom: f64,
    species: Option<&[u32]>,
    n_try: usize,
    rng: &mut R,
    mut energy: E,
) -> Vec<(usize, Vec<f64>)>
where
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
    R: rand::Rng + ?Sized,
{
    let Some(origin_slice) = origin.as_slice() else {
        return Vec::new();
    };
    let n = crate::catalog::cover_arm_count();
    let fivefold = crate::catalog::fivefold_arm();
    let mut book = crate::catalog::PackingBook::default();
    book.observe(origin_slice);
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for _ in 0..n_try.max(1) {
        let cover = crate::catalog::pick_leave_cover(n, rng);
        if !seen.insert(cover) {
            continue;
        }
        let start = if cover == fivefold {
            crate::soap::step_away_fivefold_measured(origin, LEAVE_RUNG_RMSD)
        } else {
            leave_packing_rung_to(
                origin,
                cover,
                rung_barrier(depth_per_atom, 0),
                references,
                species,
                None,
                &mut energy,
            )
        };
        if let Some(slice) = start.as_slice()
            && let Some(histogram) = book.histogram(slice)
        {
            out.push((cover, histogram));
        }
    }
    out
}

fn leave_packing_walk<R, S>(
    origin: ArrayView1<f64>,
    references: &[Vec<f64>],
    _species: Option<&[u32]>,
    _mobile: Option<&[usize]>,
    depth_per_atom: f64,
    relax_steps: usize,
    mut eval: R,
    mut step: S,
) -> Option<(f64, Array1<f64>, usize)>
where
    R: FnMut(ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    S: FnMut(ArrayView1<f64>, &mut dyn FnMut(ArrayView1<f64>) -> Option<f64>) -> Array1<f64>,
{
    let Some(origin_slice) = origin.as_slice() else {
        return None;
    };
    let (base, _) = eval(origin, 0);
    if !base.is_finite() {
        return None;
    }
    let ceiling = base + LEAVE_WALK_CLIMB * depth_per_atom.abs();
    let quench_after = rung_barrier(depth_per_atom, 2);
    let mut x = origin.to_owned();
    let mut above: Vec<(usize, Array1<f64>)> = Vec::new();
    for rung in 0..LEAVE_RIDGE_RUNGS.max(1) {
        let next = step(x.view(), &mut |trial| {
            let (energy, _) = eval(trial, 0);
            energy.is_finite().then_some(energy)
        });
        let moved = next
            .iter()
            .zip(x.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        if moved < 1e-8 {
            break;
        }
        let (value, _) = eval(next.view(), 0);
        if !(value.is_finite() && value <= ceiling) {
            break;
        }
        x = next;
        if value - base >= quench_after {
            above.push((rung, x.clone()));
        }
    }
    if above.is_empty() {
        return None;
    }
    let last = above.len() - 1;
    let mid = last / 2;
    let mut picks = vec![0, last];
    if mid != 0 && mid != last {
        picks.push(mid);
    }
    picks.sort_unstable();
    picks.dedup();
    let mut best: Option<(f64, Array1<f64>, usize)> = None;
    for index in picks {
        let (rung, point) = &above[index];
        let (q_energy, quenched) = with_disarmed(|| eval(point.view(), relax_steps.max(1)));
        if q_energy.is_finite()
            && let Some(trial) = quenched.as_slice()
            && crate::catalog::leaves_packing(origin_slice, trial, references)
            && best.as_ref().is_none_or(|(held, _, _)| q_energy < *held)
        {
            best = Some((q_energy, quenched, *rung));
        }
    }
    best
}

/// Walk the Leave ladder until the quench installs a packing.
///
/// The ladder is walked in barrier, not in length. Rung `k` aims at
/// [`rung_barrier`] and takes the step [`rung_rmsd`] that reaches it at the
/// measured `curvature` of the structure being left, so the same ladder is
/// the right size on a stiff cluster and on a soft one and needs no length
/// constant. `curvature` is the softest non-rigid eigenvalue from a Lanczos
/// pass ([`crate::curvature::curvature_features`]) and `depth_per_atom` is
/// the well depth the ladder is scaled in; a caller with neither falls back
/// to [`LEAVE_RUNG_RMSD`].
///
/// `quench` is the caller's relaxation, and the caller arms [`arm_leave`]
/// around it so the walk is not pulled back into the wells on file. The rung
/// that lands outside every packing on file is the Leave. `None` means the
/// ladder is spent, which is a refusal to report, not a reason to fall back
/// on a leftover-SOAP hole: a hole of the occupied packing quenches into the
/// occupied packing.
pub fn leave_packing_ladder<Q, E>(
    x: ArrayView1<f64>,
    cover_index: usize,
    references: &[Vec<f64>],
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    depth_per_atom: f64,
    rungs: usize,
    quench: Q,
    mut energy: E,
) -> Option<(f64, Array1<f64>, usize)>
where
    Q: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let starts: Vec<Array1<f64>> = (0..rungs.max(1))
        .map(|rung| {
            leave_packing_rung_to(
                x,
                cover_index,
                rung_barrier(depth_per_atom, rung),
                references,
                species,
                mobile,
                &mut energy,
            )
        })
        .collect();
    leave_packing_starts(x, &starts, references, quench)
}

/// Walk a prepared ladder of starts until one quenches outside every
/// packing on file, and keep the lowest-energy escape.
///
/// Split from [`leave_packing_ladder`] because building the rungs needs a
/// gradient and walking them needs a quench, and a caller whose budget
/// ledger backs both cannot hold the two closures at once.
pub fn leave_packing_starts<Q>(
    x: ArrayView1<f64>,
    starts: &[Array1<f64>],
    references: &[Vec<f64>],
    mut quench: Q,
) -> Option<(f64, Array1<f64>, usize)>
where
    Q: FnMut(ArrayView1<f64>) -> (f64, Array1<f64>),
{
    let origin = x.as_slice()?;
    let mut best: Option<(f64, Array1<f64>, usize)> = None;
    let mut spare = LEAVE_RUNG_EXTRA;
    for (rung, start) in starts.iter().enumerate() {
        let (_, walked) = quench(start.view());
        // The walk ends on a ridge, so what names the packing is the raw
        // minimum below it, not the geometry the walk stopped at. Measured
        // from LJ75 ico: the walk crosses the grain on every rung and the
        // polish returns to the floor it started on.
        let (energy, quenched) = with_disarmed(|| quench(walked.view()));
        if energy.is_finite()
            && let Some(trial) = quenched.as_slice()
            && crate::catalog::leaves_packing(origin, trial, references)
        {
            if best.as_ref().is_none_or(|(held, _, _)| energy < *held) {
                best = Some((energy, quenched, rung));
            }
            if spare == 0 {
                break;
            }
            spare -= 1;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::SeedableRng;

    #[test]
    fn unarmed_effective_is_the_raw_surface() {
        let x = Array1::from(vec![1.0, 0.0, 0.0]);
        let g = Array1::from(vec![2.0, 0.0, 0.0]);
        let (e, gt) = effective(x.view(), 3.0, g.clone());
        assert_eq!(e, 3.0);
        assert_eq!(gt, g);
    }

    #[test]
    fn shs_av_starts_pick_the_downward_distortion() {
        let x = Array1::zeros(6);
        let starts = shs_av_starts(x.view(), &[0], 0.2, 8, 1, |trial| Some(trial[0]));
        assert_eq!(starts.len(), 1);
        assert!(
            starts[0][0] < -0.05,
            "ADD channel must lower the coordinate that carries the energy, got {}",
            starts[0][0]
        );
    }

    #[test]
    fn farthest_packing_cover_is_in_range() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let index = farthest_packing_cover(x.view(), &[], 7);
        assert!(index < 7);
    }

    #[test]
    fn leave_av_walk_refuses_a_high_energy_leave() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let found = leave_av_walk(origin.view(), 1.6, 4, 0.0, &mut rng, |trial| {
            let e = trial.iter().map(|v| v * v).sum::<f64>();
            (e + 10.0, trial.to_owned())
        });
        assert!(
            found.is_none(),
            "a landing above the origin energy must not be installed"
        );
    }

    #[test]
    fn leave_av_surface_moves_the_leftover_atom() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.87, 0.0, 0.5, 0.29, 0.82, 4.0, 0.0, 0.0,
        ]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let start = leave_av_surface(x.view(), &[4], 1.6, &mut rng);
        let moved = (0..3)
            .map(|k| {
                let d = start[12 + k] - x[12 + k];
                d * d
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            moved > 0.1,
            "leftover atom must be relocated, moved={moved}"
        );
    }

    #[test]
    fn afir_term_is_the_weighted_mean_distance() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (value, grad) = afir_term(x.view(), &[0], &[1], 1.0, 1.0, &[1.0, 1.0])
            .expect("two-atom AFIR is defined");
        assert!(
            (value - 2.0).abs() < 1e-12,
            "ω cancels for one pair, V must be r, got {value}"
        );
        assert!(
            grad[0] < 0.0 && grad[3] > 0.0,
            "push gradient must point the atoms together, g0={} g1x={}",
            grad[0],
            grad[3]
        );
    }

    #[test]
    fn afir_push_and_peel_move_opposite_ways() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let origin = x.clone();
        let starts = afir_av_starts(x.view(), &[0], 0.2, 16, None, |trial| {
            let mut g = Array1::zeros(trial.len());
            let mut e = 0.0;
            for i in 0..trial.len() {
                let d = trial[i] - origin[i];
                e += 0.5 * d * d;
                g[i] = d;
            }
            Some((e, g))
        });
        assert_eq!(starts.len(), 2);
        let pair = |state: &Array1<f64>| {
            let dx = state[0] - state[3];
            let dy = state[1] - state[4];
            let dz = state[2] - state[5];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let push = pair(&starts[0]);
        let peel = pair(&starts[1]);
        assert!(
            push < 2.0 && peel > 2.0,
            "push must compress and peel must open, push={push} peel={peel}"
        );
    }

    #[test]
    fn householder_flips_a_gradient_that_points_at_the_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        // x is to the +x of the well; g points +x, so descent (-g) walks
        // toward the well.
        let x = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (_, gt) = effective(x.view(), 1.0, g);
        assert!(
            gt[0] < 0.0,
            "transformed gradient must point away from the well, g0={}",
            gt[0]
        );
        disarm();
    }

    #[test]
    fn hill_raises_energy_near_a_known_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        let x = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (e, _) = effective(x.view(), 0.0, g);
        assert!(e > 0.0, "Gaussian hill must raise E, got {e}");
        disarm();
    }

    #[test]
    fn rgmin_on_the_transformed_surface_walks_away_from_the_well() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        let start = Array1::from(vec![0.3, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (_, x) = step_rgmin(&mut opt, start.view(), 40, |v| {
            // Harmonic well at `origin`. Raw descent walks back.
            let mut g = Array1::zeros(v.len());
            let mut e = 0.0;
            for i in 0..v.len() {
                let d = v[i] - origin[i];
                e += 0.5 * d * d;
                g[i] = d;
            }
            Some(effective(v, e, g))
        });
        let (_, start_rmsd) = com_free_delta(start.view(), origin.view());
        let (_, end_rmsd) = com_free_delta(x.view(), origin.view());
        disarm();
        assert!(
            end_rmsd > start_rmsd,
            "rgmin must walk away from the known well, start={start_rmsd} end={end_rmsd}"
        );
    }

    #[test]
    fn arrivals_raise_the_cost_of_leaving_a_packing() {
        // Two clouds holding the same well. The second has been arrived
        // on a hundred times, so its free-energy depth exceeds the first
        // by T ln 100, and a Leave from it has that much more to pay.
        let origin = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let once = vec![crate::catalog::PackingReference {
            coordinates: origin.to_vec(),
            visits: 1,
            deposit: 0.0,
        }];
        let often = vec![crate::catalog::PackingReference {
            coordinates: origin.to_vec(),
            visits: 100,
            deposit: 0.0,
        }];
        let rare = {
            arm_leave_free(origin.view(), 0.35, &once, 0.8, 0.0);
            let entropy = ARMED.with(|slot| slot.borrow().as_ref().map(|a| a.wells[0].entropy));
            disarm();
            entropy.expect("armed")
        };
        let common = {
            arm_leave_free(origin.view(), 0.35, &often, 0.8, 0.0);
            let entropy = ARMED.with(|slot| slot.borrow().as_ref().map(|a| a.wells[0].entropy));
            disarm();
            entropy.expect("armed")
        };
        assert_eq!(rare, 0.0, "one arrival is zero entropy, got {rare}");
        let want = 0.8 * 100.0_f64.ln();
        assert!(
            (common - want).abs() < 1e-12,
            "T ln n must price the packing, want {want} got {common}"
        );
    }

    #[test]
    fn a_standing_deposit_shortens_the_next_hill() {
        // Well-tempered scaling. A well already carrying dT of bias takes
        // a shorter hill than a fresh one, so the pile converges instead
        // of growing without bound.
        //
        // On a real cluster, because the free-energy deposit lives on the
        // packing path: a structure with no nu=3 mean falls through to
        // the Cartesian transform, which carries no entropy and no
        // tempering, and a two-atom toy reports the same hill either way
        // whatever is standing on it.
        let origin = ico13();
        let mut x = origin.clone();
        x[3] += 0.25;
        let g = Array1::from(vec![0.05; origin.len()]);
        let fresh = vec![crate::catalog::PackingReference {
            coordinates: origin.to_vec(),
            visits: 1,
            deposit: 0.0,
        }];
        let standing = vec![crate::catalog::PackingReference {
            coordinates: origin.to_vec(),
            visits: 1,
            deposit: 0.8,
        }];
        arm_leave_free(origin.view(), 0.35, &fresh, 0.8, 0.8);
        let (full, _) = effective(x.view(), 0.0, g.clone());
        disarm();
        arm_leave_free(origin.view(), 0.35, &standing, 0.8, 0.8);
        let (tempered, _) = effective(x.view(), 0.0, g);
        disarm();
        assert!(full > 0.0, "a fresh well takes a full hill, got {full}");
        assert!(
            tempered < full,
            "a well already carrying bias takes a shorter hill, full={full} tempered={tempered}"
        );
        // e is the scale, not an arbitrary shrink: the deposit equals dT.
        let want = full / std::f64::consts::E;
        assert!(
            (tempered - want).abs() < 1e-9 * full.max(1.0),
            "a deposit of dT must cost exactly one e-fold, want {want} got {tempered}"
        );
    }

    #[test]
    fn disarm_restores_the_raw_surface() {
        let origin = Array1::from(vec![0.0, 0.0, 0.0]);
        arm_leave(origin.view(), 0.35, &[]);
        disarm();
        let x = Array1::from(vec![0.2, 0.0, 0.0]);
        let g = Array1::from(vec![1.0, 0.0, 0.0]);
        let (e, gt) = effective(x.view(), 4.0, g.clone());
        assert_eq!(e, 4.0);
        assert_eq!(gt, g);
    }

    fn ico13() -> Array1<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        let s = 1.0 / (1.0 + p * p).sqrt();
        let mut x = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    #[test]
    fn span_rises_when_the_packing_mean_leaves_mu_k() {
        let origin = ico13();
        arm_leave(origin.view(), 0.35, &[]);
        let at_well = span(origin.view());
        let mut away = origin.clone();
        away[3] += 0.25;
        let off = span(away.view());
        disarm();
        assert!(
            off > at_well,
            "span must rise away from mu_k, well={at_well} off={off}"
        );
    }

    #[test]
    fn packing_mode_is_nu3_mean_not_leftover_soap() {
        let origin = ico13();
        let mu = packing_mean(origin.view()).expect("ico13 packing mean");
        let want = PACKING_SPEC.feat_dim(None) + PACKING_SPEC.nu3_feat_dim(None);
        assert_eq!(
            mu.len(),
            want,
            "packing invert must live in stacked SOAP+ν=3, not leftover SOAP"
        );
    }

    #[test]
    fn packing_householder_flips_force_along_the_nu3_pullback() {
        let origin = ico13();
        arm_leave(origin.view(), 0.35, &[]);
        let mut x = origin.clone();
        x[3] += 0.25;
        let mu0 = packing_mean(origin.view()).expect("origin packing mean");
        let (p, r_phi, _) = packing_pullback(x.view(), mu0.view()).expect("packing mode");
        assert!(r_phi > 0.0, "displaced ico13 must leave the origin μ");
        let (_, gt) = effective(x.view(), 1.0, p.clone());
        let proj = gt.iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f64>();
        disarm();
        assert!(proj < 0.0, "packing Householder must flip g·P, got {proj}");
    }

    #[test]
    fn neighbour_invert_freezes_p_at_arm() {
        let origin = ico13();
        let mut other = origin.clone();
        other[0] += 0.2;
        arm_leave_free(
            origin.view(),
            0.35,
            &[crate::catalog::PackingReference {
                coordinates: other.to_vec(),
                visits: 1,
                deposit: 0.0,
            }],
            0.8,
            0.8,
        );
        assert!(is_armed());
        assert!(
            invert_is_frozen(),
            "neighbour invert must lift P once at arm"
        );
        disarm();
    }

    #[test]
    fn rgmin_walks_off_the_known_packing() {
        let origin = ico13();
        arm_leave(origin.view(), 0.35, &[]);
        let mut start = origin.clone();
        start[3] += 0.20;
        let mu0 = packing_mean(origin.view()).expect("origin packing mean");
        let start_span = packing_mean(start.view())
            .map(|mu| packing_l2(mu.view(), mu0.view()))
            .expect("start packing mean");
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (_, x) = step_rgmin(&mut opt, start.view(), 16, |v| {
            let mut g = Array1::zeros(v.len());
            let mut e = 0.0;
            for i in 0..v.len() {
                let d = v[i] - origin[i];
                e += 0.5 * d * d;
                g[i] = d;
            }
            Some(effective(v, e, g))
        });
        let end_span = packing_mean(x.view())
            .map(|mu| packing_l2(mu.view(), mu0.view()))
            .unwrap_or(0.0);
        disarm();
        assert!(
            end_span > start_span,
            "rgmin must walk off the known packing, start={start_span} end={end_span}"
        );
    }
}
