//! Running a cluster search against an objective.
//!
//! [`crate::methods::cluster_hopping::run`] takes a relaxation and a gradient as
//! closures, which is the right interface for a driver: it does not care where
//! the energy comes from. It is the wrong interface for a caller, because every
//! caller then writes the same three things, and the campaign this crate
//! reports was run against potentials defined inside its own examples.
//!
//! This is the missing half. Hand it anything implementing
//! [`DifferentiableObjective<f64>`] and it builds the relaxation, charges every
//! evaluation to the ledger, counts what converged and runs the search.
//!
//! What that buys is provenance. rgpot reaches this crate as an
//! `eindir_objective_t`, wrapped into an `Objective<f64>`, so a potential from
//! there arrives at the cluster driver by the same route as one written here
//! and neither the driver nor this function can tell them apart.

use crate::methods::bond_preconditioner::BondPreconditioner;
use crate::methods::cluster_hopping::OVERLAP_SEPARATION;
use crate::methods::cluster_hopping::min_pair_distance;
#[cfg(any(feature = "bank-rpc", test))]
use crate::methods::cluster_hopping::structure_is_sane;
use crate::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, Ledger, Outcome, SearchError,
    optimize_with_energy_gradient, run_with_energy_gradient_at_checkpoints,
};
use crate::methods::warm_lbfgs::WarmLbfgs;
use crate::quench::{QuenchPredictor, Verdict};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::Mutex;

/// What a search did, beyond the outcome the driver reports.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RelaxStats {
    /// Relaxations that reached a point with a small gradient.
    pub converged: usize,
    /// Charged evaluations spent in screening passes.
    ///
    /// Split from the full relaxations because the two are different levers.
    /// Every mechanism in this crate that tried to change where the chain goes
    /// was measured and failed; the one that helped, the return screen, buys
    /// hops by not paying for relaxations that will be discarded. If throughput
    /// is what moves the number then knowing which pass the budget goes to is
    /// the first thing to establish, and it has never been measured here.
    pub screen_charged: usize,
    /// Charged evaluations spent in full relaxations.
    pub full_charged: usize,
    /// Charged evaluations spent confirming convergence.
    pub check_charged: usize,
    /// Relaxations that stopped at their iteration cap.
    ///
    /// A large share of these is not by itself wrong, because the screening
    /// pass is capped deliberately, but a run where nothing converges is not on
    /// the quenched landscape and every mechanism above it is acting on noise.
    pub capped: usize,
    /// Screening passes run.
    pub screens: usize,
    /// Screens where the predictor would have stopped, under probing.
    pub probe_stops: usize,
    /// Steps at which it would have stopped, summed.
    pub probe_steps: usize,
    /// Absolute error of the extrapolation against the full screen, summed.
    ///
    /// The number that decides whether a screening pass can be shortened at
    /// all. If the extrapolation from five steps predicts the twenty-five step
    /// energy to well inside the spacing between neighbouring minima, the extra
    /// twenty steps are buying precision nothing uses. If it does not, the
    /// screen is not overhead around the quench, it is the quench.
    pub probe_error: f64,
    /// Line-search geometries closer than the overlap rule.
    ///
    /// Each one is refused before the potential is called, and it is not
    /// included in [`RelaxStats::charged`].
    pub overlap_rejected: usize,
    /// Descent steps those passes took, summed.
    ///
    /// Against `screens * screen_steps` this is what stopping on a decision
    /// bought, and it is the only number that says whether it bought anything.
    pub screen_steps_taken: usize,
}

impl RelaxStats {
    /// Relaxation calls made.
    pub fn total(&self) -> usize {
        self.converged + self.capped
    }

    /// Charged evaluations across both passes and the convergence check.
    pub fn charged(&self) -> usize {
        self.screen_charged + self.full_charged + self.check_charged
    }

    /// Share of the charged budget spent screening.
    pub fn screen_share(&self) -> f64 {
        let t = self.charged();
        if t == 0 {
            return 0.0;
        }
        self.screen_charged as f64 / t as f64
    }
}

/// Gradient magnitude below which a relaxation counts as converged.
///
/// Loose enough that a screening pass is not called converged and tight enough
/// that a genuine minimum is: on a Lennard-Jones cluster a quenched structure
/// comes back at about 1e-6.
pub const CONVERGED_GRADIENT: f64 = 1e-3;

fn two_phase_penalty(
    x: ArrayView1<f64>,
    groups: Option<&[Vec<usize>]>,
    cutoff: f64,
    beta: f64,
    mu: f64,
    shape: Option<&crate::methods::two_phase::Shape>,
) -> (f64, Array1<f64>) {
    match groups.filter(|declared| !declared.is_empty()) {
        Some(groups) => crate::methods::two_phase::penalty_groups(x, groups, cutoff, beta, mu),
        None => crate::methods::two_phase::penalty_shaped(x, cutoff, beta, mu, shape),
    }
}

fn gradient_is_converged(gradient: ArrayView1<f64>, threshold: f64) -> bool {
    // `f64::max` returns its other argument against NaN, so an infinity-norm
    // fold of an all-NaN gradient is zero and a failed evaluation is scored
    // as converged. A failed call is not a small gradient.
    !gradient.is_empty()
        && gradient.iter().all(|value| value.is_finite())
        && largest_component(gradient) < threshold
}

/// The convergence check a relaxation paid for at the point it ended.
#[derive(Clone)]
struct Check {
    at: Array1<f64>,
    energy: f64,
    gradient: Array1<f64>,
}

type PaidCheck = Mutex<Option<Check>>;

fn largest_component(gradient: ArrayView1<f64>) -> f64 {
    gradient
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
}

/// The gradient driver's record gate, reusing the relaxation's paid check.
///
/// The driver validates a quench by asking for the gradient at the point the
/// relaxation returned, and the relaxation has just paid for that gradient.
/// Charging it twice spends a potential call on an answer already in hand.
/// Any other point is evaluated and charged as before. A non-finite answer
/// is not handed to the driver.
fn gated_grad<'a, O>(
    objective: &'a O,
    paid: &'a PaidCheck,
) -> impl FnMut(&mut Ledger, ArrayView1<f64>) -> Option<Array1<f64>> + 'a
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    move |led, x| {
        let reused = paid
            .lock()
            .expect("paid check")
            .as_ref()
            .filter(|check| check.at == x)
            .map(|check| check.gradient.clone());
        if reused.is_some() {
            return reused;
        }
        if !led.charge() {
            return None;
        }
        finite_gradient(objective, x)
    }
}

fn finite_pair(dim: usize, energy: f64, gradient: &Array1<f64>) -> bool {
    energy.is_finite() && gradient.len() == dim && gradient.iter().all(|value| value.is_finite())
}

/// Uphill stand-in for a non-finite evaluation.
///
/// Large enough that a line search will not accept it over a physical
/// energy, and steep enough that its infinity norm cannot meet a record
/// tolerance. The search shortens that step and can still reach a finite
/// quench. Without an anchor there is no finite point to return to.
const FAILED_STEP_ENERGY: f64 = 1.0e6;
const FAILED_STEP_SLOPE: f64 = 1.0;

fn failed_step(point: ArrayView1<f64>, anchor: Option<&Array1<f64>>) -> Option<(f64, Array1<f64>)> {
    let anchor = anchor?;
    if anchor.len() != point.len() || point.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut gradient = Array1::zeros(point.len());
    let mut peak = 0.0_f64;
    for index in 0..point.len() {
        let delta = point[index] - anchor[index];
        gradient[index] = delta;
        peak = peak.max(delta.abs());
    }
    if !(peak >= FAILED_STEP_SLOPE) {
        let index = gradient
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.abs()
                    .partial_cmp(&right.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let sign = if gradient[index] < 0.0 { -1.0 } else { 1.0 };
        gradient[index] = sign * FAILED_STEP_SLOPE;
    }
    Some((FAILED_STEP_ENERGY, gradient))
}

/// Physical `(energy, gradient)`, or an uphill rejection of a non-finite answer.
///
/// The anchor is the last finite geometry. A later failure steps back toward
/// it instead of becoming the quench result.
fn evaluated_pair<O>(
    objective: &O,
    point: ArrayView1<f64>,
    anchor: &mut Option<Array1<f64>>,
) -> Option<(f64, Array1<f64>, bool)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (energy, gradient) = objective.value_and_gradient(point);
    if finite_pair(point.len(), energy, &gradient) {
        *anchor = Some(point.to_owned());
        return Some((energy, gradient, true));
    }
    failed_step(point, anchor.as_ref()).map(|(energy, gradient)| (energy, gradient, false))
}

fn finite_gradient<O>(objective: &O, x: ArrayView1<f64>) -> Option<Array1<f64>>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let gradient = objective.grad(x);
    (gradient.len() == x.len() && gradient.iter().all(|value| value.is_finite()))
        .then_some(gradient)
}

/// Uphill stand-in for a geometry the overlap rule refuses.
///
/// The potential is not called. The line search still receives a step it
/// will shorten, so a later trial can land on a separated geometry.
fn overlap_rejection_step(
    point: ArrayView1<f64>,
    anchor: Option<&Array1<f64>>,
) -> Option<(f64, Array1<f64>)> {
    if let Some(step) = failed_step(point, anchor) {
        return Some(step);
    }
    if point.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut gradient = Array1::zeros(point.len());
    if let Some(value) = gradient.first_mut() {
        *value = FAILED_STEP_SLOPE;
    }
    Some((FAILED_STEP_ENERGY, gradient))
}

/// Charge a physical evaluation, or refuse an overlap without charging it.
fn charge_or_refuse_overlap<O>(
    objective: &O,
    led: &mut Ledger,
    stats: &mut RelaxStats,
    point: ArrayView1<f64>,
    anchor: &mut Option<Array1<f64>>,
    refuse_overlap: bool,
) -> Option<(f64, Array1<f64>, bool)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    if refuse_overlap && min_pair_distance(point) < OVERLAP_SEPARATION {
        stats.overlap_rejected += 1;
        return overlap_rejection_step(point, anchor.as_ref())
            .map(|(energy, gradient)| (energy, gradient, false));
    }
    if !led.charge() {
        return None;
    }
    evaluated_pair(objective, point, anchor)
}

fn warm_relax<'a, O>(
    objective: &'a O,
    cfg: &'a Config,
    stats: &'a mut RelaxStats,
    seed: u64,
    paid: &'a PaidCheck,
) -> impl FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + 'a
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut opt = WarmLbfgs::default();
    // A minimum is recordable at `record_gradient`; descending past it buys
    // digits no acceptance test or record reads. Stopping at 1e-6 instead cost
    // the communicating preset a PAR-2 of 5792 force calls on 32 LJ38 starts
    // against 4269 here, and a median of 5726 against 3253. Rigid bodies and
    // species diameters record at that same tolerance.
    //
    // A record polish is the exception. Its answer is the energy a run
    // reports, and an energy at the gate is not settled. Take 16 bests of a
    // 42-atom peptide at GFN2-xTB, stopped at 5e-2 eV/Å. The xtb optimiser
    // lowered them by 2.07 kcal/mol on average. The polish descends to the
    // solver's own default.
    let settled = opt.gtol;
    let stop = cfg.record_gradient;
    // The driver calls the screening pass with `screen_steps`, the full one
    // with `relax_steps` and a record polish with `polish_records`, so the
    // iteration count identifies which is which.
    let screen_iters = cfg.screen_steps;
    let polish_iters = cfg.polish_records;
    let adaptive = cfg.adaptive_screen;
    let probe = cfg.probe_screen;
    let species = cfg
        .species
        .as_deref()
        .filter(|_| cfg.precondition_bonds)
        .filter(|z| z.len() == cfg.n_points);
    let mut surfaces = (!cfg.surfaces.is_empty())
        .then(|| crate::methods::two_phase::SurfacePortfolio::new(&cfg.surfaces, seed));
    let mut anchor: Option<Array1<f64>> = None;
    let refuse_overlap = cfg.species.is_some();
    move |led, x, iters| {
        opt.forget();
        let polishing = polish_iters > 0
            && iters == polish_iters
            && iters != cfg.relax_steps
            && iters > screen_iters;
        opt.gtol = if polishing { settled } else { stop };
        let prior = paid
            .lock()
            .expect("paid check")
            .take()
            .filter(|check| check.at == x);
        let before = led.spent();
        let screening = iters <= screen_iters;
        // A rigid chart uses this same descent. A line-search-free capped
        // step reached a different basin, and the small-cluster walk then
        // took more hops at the same cost per quench.
        //
        // A start the previous relaxation already certified at or below the
        // stop has nothing left to descend, and the descent's first call
        // would pay again for the energy and gradient on file.
        let plain = surfaces.is_none() && !cfg.two_phase.is_some_and(|two| two.is_active());
        if plain
            && let Some(check) = prior.as_ref()
            && largest_component(check.gradient.view()) <= opt.gtol
            && gradient_is_converged(check.gradient.view(), opt.gtol)
        {
            if gradient_is_converged(check.gradient.view(), cfg.record_gradient) {
                stats.converged += 1;
            } else {
                stats.capped += 1;
            }
            let energy = check.energy;
            *paid.lock().expect("paid check") = prior;
            return (energy, x.to_owned());
        }
        // The screening pass stops when its own trajectory says the question is
        // settled; the full relaxation runs to its tolerance, because there the
        // answer is the structure and not a verdict about it.
        let mut pred = QuenchPredictor::new();
        pred.warmup = cfg.quench_warmup;
        pred.confidence = cfg.quench_confidence;
        let mut early = false;
        let mut probe_at: Option<(usize, f64)> = None;
        let target = led.best;
        // Phase one, when configured: relax on the compacted surface and hand
        // that minimum to the plain relaxation below. Both phases charge.
        let compacted;
        let surface = match surfaces.as_mut() {
            Some(portfolio) => portfolio.begin(screening),
            None => cfg.two_phase.filter(|two| two.is_active()),
        };
        // The last physical point the descent paid for, kept so the
        // convergence check below does not pay for it again. A synthetic
        // uphill rejection is not that point.
        let last: RefCell<Option<Check>> = RefCell::new(None);
        let remember =
            |point: ArrayView1<f64>, energy: f64, gradient: &Array1<f64>, physical: bool| {
                if physical {
                    *last.borrow_mut() = Some(Check {
                        at: point.to_owned(),
                        energy,
                        gradient: gradient.clone(),
                    });
                } else {
                    *last.borrow_mut() = None;
                }
            };
        let x = match surface {
            Some(two) => {
                let cutoff = two.cutoff_for(x);
                let shape = two.shape_for(x);
                let (_, phase_one, _) = opt.minimize(x, iters, |v| {
                    let (e, g, physical) = charge_or_refuse_overlap(
                        objective,
                        led,
                        stats,
                        v,
                        &mut anchor,
                        refuse_overlap,
                    )?;
                    remember(v, e, &g, physical);
                    if !physical {
                        return Some((e, g));
                    }
                    let (pe, pg) = two_phase_penalty(
                        v,
                        cfg.move_library.declared_groups(),
                        cutoff,
                        two.beta,
                        two.mu,
                        shape.as_ref(),
                    );
                    Some((e + pe, g + pg))
                });
                opt.forget();
                compacted = phase_one;
                compacted.view()
            }
            None => x,
        };
        let stop_at = led.target();
        // A descent from where the previous one ended, the full pass after
        // a screen among them, opens by asking for an answer already paid.
        let mut opening = prior.clone();
        let mut evaluate = |v: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
            if let Some(check) = opening.take_if(|check| check.at == v) {
                let answer = (check.energy, check.gradient.clone());
                *last.borrow_mut() = Some(check);
                return Some(answer);
            }
            let (energy, gradient, physical) =
                charge_or_refuse_overlap(objective, led, stats, v, &mut anchor, refuse_overlap)?;
            remember(v, energy, &gradient, physical);
            Some((energy, gradient))
        };
        let mut watch = |_: usize, fv: f64| {
            if stop_at.is_some_and(|limit| fv <= limit) {
                return false;
            }
            if screening && probe {
                // Never stops. Records where a stop would have happened and
                // what it would have claimed, so the claim can be scored
                // against the value the full pass actually reaches.
                pred.observe(fv);
                if probe_at.is_none() && pred.verdict(target) == Verdict::Hopeless {
                    probe_at = pred.predict().map(|p| (pred.len(), p.limit));
                }
                return true;
            }
            if !(screening && adaptive) {
                return true;
            }
            pred.observe(fv);
            // Only the hopeless verdict stops the descent. A promising one
            // is followed by the full relaxation anyway, so cutting its
            // screen short saves at most the tail of a pass that 2 per cent
            // of trials reach, and it hands the return screen an
            // extrapolated energy below the incumbent attached to a
            // structure that was never relaxed.
            if pred.verdict(target) != Verdict::Hopeless {
                return true;
            }
            early = true;
            false
        };
        let preconditioner = species.and_then(|z| BondPreconditioner::new(x, z));
        let (f, xr) = match preconditioner {
            None => {
                let (f, xr, _) = opt.minimize_watched(x, iters, &mut evaluate, &mut watch);
                (f, xr)
            }
            Some(pre) => {
                // The reduced gradient has other units, so the solver's own
                // test is off and the stop reads the Cartesian one.
                let origin = x.to_owned();
                let gtol = opt.gtol;
                opt.gtol = 0.0;
                let (f, zr, _) = opt.minimize_watched(
                    Array1::zeros(x.len()).view(),
                    iters,
                    |z| {
                        let v = pre.expand(origin.view(), z);
                        evaluate(v.view()).map(|(e, g)| (e, pre.reduce(g.view())))
                    },
                    |it, fv| {
                        let settled = last
                            .borrow()
                            .as_ref()
                            .is_some_and(|c| largest_component(c.gradient.view()) <= gtol);
                        !settled && watch(it, fv)
                    },
                );
                opt.gtol = gtol;
                (f, pre.expand(origin.view(), zr.view()))
            }
        };
        if let Some((at, claim)) = probe_at {
            stats.probe_stops += 1;
            stats.probe_steps += at;
            stats.probe_error += (claim - f).abs();
        }
        let cost = led.spent() - before;
        if screening {
            stats.screen_charged += cost;
            stats.screen_steps_taken += pred.len();
            stats.screens += 1;
        } else {
            stats.full_charged += cost;
        }
        // The energy the caller sees is the extrapolated limit, not the value
        // at the point where the descent stopped.
        //
        // This is the whole of it. Stopping a screening quench after five steps
        // instead of twenty-five cut the cost of a hop from 31 charged
        // evaluations to 8 and quadrupled the hops, and solved nothing in three
        // seeds where the fixed-length screen solved three: the value at step
        // five is not the quenched energy, and a chain that accepts on it is
        // walking on the raw landscape rather than the transformed one that
        // basin hopping exists to walk on. The predictor already says where the
        // descent was going; the estimate is what the chain should move on.
        //
        // The floor in `stopped_energy` is what keeps a cut-short descent from
        // being reported as the run's answer. It is enforced there rather than
        // inferred from the verdict here.
        let f = if early {
            pred.stopped_energy(target, f)
        } else {
            f
        };
        if let Some(portfolio) = surfaces.as_mut() {
            portfolio.observe(screening, f, target);
        }
        // A descent stopped on a verdict is known not to be converged, so
        // asking is spending an evaluation on an answer already in hand. The
        // check stays on every other path, where the answer is not known.
        if early {
            stats.capped += 1;
            // The water census reads these boundaries. A screened descent is
            // not a minimum, so it is recorded without a validation gradient.
            led.record_quench_boundary(before, f, xr.clone(), None);
            *paid.lock().expect("paid check") = last.borrow().clone();
            return (f, xr);
        }
        // Charged like any other evaluation: asking whether a relaxation
        // converged is asking the potential a question, and a protocol that
        // counts evaluations has to count this one.
        // A descent that never left a certified start already has its answer,
        // and so does one whose last paid evaluation is where it stopped.
        let mut f = f;
        let mut xr = xr;
        let stored_gradient = |point: &Array1<f64>| -> Option<Array1<f64>> {
            last.borrow()
                .clone()
                .filter(|check| &check.at == point)
                .or_else(|| prior.clone().filter(|check| &check.at == point))
                .map(|check| check.gradient)
        };
        let mut validated = None;
        let confirm = |led: &mut Ledger,
                       stats: &mut RelaxStats,
                       energy: f64,
                       point: ArrayView1<f64>|
         -> Option<Array1<f64>> {
            if refuse_overlap && min_pair_distance(point) < OVERLAP_SEPARATION {
                stats.overlap_rejected += 1;
                return None;
            }
            if !energy.is_finite() {
                return None;
            }
            if let Some(gradient) = stored_gradient(&point.to_owned()) {
                return gradient_is_converged(gradient.view(), cfg.record_gradient)
                    .then_some(gradient);
            }
            if !led.charge() {
                return None;
            }
            stats.check_charged += 1;
            finite_gradient(objective, point)
                .filter(|gradient| gradient_is_converged(gradient.view(), cfg.record_gradient))
        };
        let mut converged = match confirm(led, stats, f, xr.view()) {
            Some(gradient) => {
                validated = Some(gradient);
                true
            }
            None => false,
        };
        // A capped descent is not a minimum the ledger may keep. One more
        // pass is enough for a binary quench. A molecular potential, whose
        // line search otherwise stops inside an overlap, is continued until
        // the same record gradient is met or the budget ends. Screening
        // passes stay short, and the tolerance is the preset's own.
        let continuation_rounds = if screening {
            0
        } else if cfg.species.is_some() && cfg.species_sigma.is_none() {
            8
        } else if cfg.species_sigma.is_some() {
            1
        } else {
            0
        };
        for _ in 0..continuation_rounds {
            if converged || led.remaining() == 0 {
                break;
            }
            let spent_at = led.spent();
            let (f2, xr2, _) = opt.minimize(xr.view(), iters, |v| {
                let (energy, gradient, physical) = charge_or_refuse_overlap(
                    objective,
                    led,
                    stats,
                    v,
                    &mut anchor,
                    refuse_overlap,
                )?;
                remember(v, energy, &gradient, physical);
                Some((energy, gradient))
            });
            let extra = led.spent() - spent_at;
            if extra > 0 {
                stats.full_charged += extra;
            }
            f = f2;
            xr = xr2;
            converged = match confirm(led, stats, f, xr.view()) {
                Some(gradient) => {
                    validated = Some(gradient);
                    true
                }
                None => {
                    validated = None;
                    false
                }
            };
        }
        if converged {
            stats.converged += 1;
        } else {
            stats.capped += 1;
            validated = None;
        }
        // Good-Turing retires from the checkpoint's quench list. Leaving this
        // empty made that stop observe nothing and run out the budget.
        led.record_quench_boundary(before, f, xr.clone(), validated.clone());
        *paid.lock().expect("paid check") = validated
            .map(|gradient| Check {
                at: xr.clone(),
                energy: f,
                gradient,
            })
            .or_else(|| last.borrow().clone().filter(|check| check.at == xr));
        (f, xr)
    }
}

/// Runs a cluster search on `objective` under `ledger`.
///
/// The relaxation is this crate's warm-started quasi-Newton one, and its
/// curvature is deliberately not carried between calls: measured on a cluster,
/// retaining it across a structural change costs more than it saves.
pub fn search<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    seed: u64,
) -> Result<(Outcome, RelaxStats), SearchError>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut stats = RelaxStats::default();

    // Split deliberately: the relaxation needs the optimizer mutably, the
    // gradient needs only the objective. Sharing the objective by reference is
    // what lets both closures exist at once.
    let out = {
        let paid = PaidCheck::default();
        let mut relax = warm_relax(objective, cfg, &mut stats, seed, &paid);
        let mut grad = gated_grad(objective, &paid);

        // Value and gradient in one charge, which is what a Hamiltonian leapfrog
        // leaf needs. On a pairwise potential both come out of one pass over the
        // pairs, so charging twice would understate the arm by a factor of two.
        // A non-finite answer is an uphill rejection, not a step the leaf can adopt.
        let mut energy_anchor = None;
        let mut energy_grad =
            |led: &mut Ledger, x: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
                if !led.charge() {
                    return None;
                }
                evaluated_pair(objective, x, &mut energy_anchor)
                    .map(|(energy, gradient, _)| (energy, gradient))
            };

        optimize_with_energy_gradient(
            cfg,
            ledger,
            &mut relax,
            Some(&mut grad),
            Some(&mut energy_grad),
            seed,
        )?
    };
    Ok((out, stats))
}

/// As [`search`], from a geometry the caller already has.
///
/// A slab or a packed molecular start is not a random cluster in a sphere.
/// The hop RNG is seeded independently of that geometry.
pub fn search_from<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    seed: u64,
) -> Result<(Outcome, RelaxStats), SearchError>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut stats = RelaxStats::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let out = {
        let paid = PaidCheck::default();
        let mut relax = warm_relax(objective, cfg, &mut stats, seed, &paid);
        let mut grad = gated_grad(objective, &paid);
        let mut energy_anchor = None;
        let mut energy_grad =
            |led: &mut Ledger, x: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
                if !led.charge() {
                    return None;
                }
                evaluated_pair(objective, x, &mut energy_anchor)
                    .map(|(energy, gradient, _)| (energy, gradient))
            };
        crate::methods::cluster_hopping::run_with_energy_gradient(
            cfg,
            start,
            ledger,
            &mut relax,
            Some(&mut grad),
            Some(&mut energy_grad),
            &mut rng,
        )?
    };
    Ok((out, stats))
}

/// Relaxes a reported answer until every gradient component is below
/// `tolerance`, charging each evaluation to `ledger`.
///
/// A hop records a minimum at [`Config::record_gradient`], which ranks
/// basins but leaves the stored energy above the minimum by the residual
/// curvature term. A comparison that reads the energy to a tighter window
/// needs the answer at its own stationary-point gate. Returns `None` when
/// the ledger closes before the first evaluation.
pub fn polish_answer<O>(
    objective: &O,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    tolerance: f64,
    max_iter: usize,
) -> Option<(f64, Array1<f64>)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut opt = WarmLbfgs::default();
    opt.gtol = tolerance;
    let mut anchor = None;
    let mut evaluated = false;
    let (energy, state, _) = opt.minimize(start, max_iter, |v| {
        if !ledger.charge() {
            return None;
        }
        evaluated = true;
        evaluated_pair(objective, v, &mut anchor).map(|(energy, gradient, _)| (energy, gradient))
    });
    (evaluated && energy.is_finite()).then_some((energy, state))
}

/// [`search_from`] with a checkpoint after each block of charged work.
///
/// The water Good-Turing arm retires from this hook once the census of
/// validated quenches is saturated. Other arms pass a checkpoint that
/// continues.
pub fn search_from_with_checkpoint<O, H>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    seed: u64,
    checkpoint_interval: usize,
    checkpoint: &mut H,
) -> Result<(Outcome, RelaxStats), SearchError>
where
    O: DifferentiableObjective<f64> + ?Sized,
    H: for<'a> FnMut(ChainCheckpoint<'a>) -> CheckpointAction,
{
    let mut stats = RelaxStats::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let out = {
        let paid = PaidCheck::default();
        let mut relax = warm_relax(objective, cfg, &mut stats, seed, &paid);
        let mut grad = gated_grad(objective, &paid);
        let mut energy_anchor = None;
        let mut energy_grad =
            |led: &mut Ledger, x: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
                if !led.charge() {
                    return None;
                }
                evaluated_pair(objective, x, &mut energy_anchor)
                    .map(|(energy, gradient, _)| (energy, gradient))
            };
        run_with_energy_gradient_at_checkpoints(
            cfg,
            start,
            ledger,
            &mut relax,
            Some(&mut grad),
            Some(&mut energy_grad),
            &mut rng,
            checkpoint_interval,
            checkpoint,
        )?
    };
    Ok((out, stats))
}

/// [`search_from`] when `BANK_RPC` is unset; [`search_from_bank`] when it is.
///
/// One binary, two arms. The control is the same walk without the shared
/// catalog. First-encounter charged evaluations are what says which is
/// cheaper, not whether both finished.
pub fn search_from_maybe_bank<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    seed: u64,
) -> Result<(Outcome, RelaxStats), SearchError>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    match std::env::var("BANK_RPC") {
        Ok(sock) if !sock.is_empty() => {
            #[cfg(feature = "bank-rpc")]
            {
                search_from_bank(objective, cfg, ledger, start, seed, &sock)
            }
            #[cfg(not(feature = "bank-rpc"))]
            {
                let _ = sock;
                panic!("BANK_RPC set; rebuild with --features bank-rpc");
            }
        }
        _ => search_from(objective, cfg, ledger, start, seed),
    }
}

/// Keep adsorbate atoms above the frozen slab after a SOAP hole step.
#[cfg(all(feature = "bank-rpc", feature = "featomic"))]
fn pin_adsorbate_above_slab(x: &mut Array1<f64>, cfg: &Config) {
    let Some((seeds, _)) = cfg.active_region.as_ref() else {
        return;
    };
    let n = x.len() / 3;
    let mut z_top = f64::NEG_INFINITY;
    for i in 0..n {
        if seeds.contains(&i) {
            continue;
        }
        z_top = z_top.max(x[3 * i + 2]);
    }
    if !z_top.is_finite() {
        return;
    }
    let floor = z_top + 0.8;
    for &i in seeds {
        if x[3 * i + 2] < floor {
            x[3 * i + 2] = floor;
        }
    }
}

/// Pair distance below which a member is an overlap catastrophe, not a bond.
///
/// H–H is 0.74 Å. EAM overlap sits under 0.3 Å. The configured
/// `min_separation` is scaled by the largest covalent diameter, which
/// on a Cu slab is copper, and that floor rejects every physical H2.
#[cfg(feature = "bank-rpc")]
fn sane_sep(_cfg: &Config) -> f64 {
    OVERLAP_SEPARATION
}

/// Mobile atom indices: the active region, or the complement of `frozen`.
#[cfg(all(feature = "bank-rpc", feature = "featomic"))]
fn mobile_of(cfg: &Config) -> Option<Vec<usize>> {
    if let Some((seeds, _)) = cfg.active_region.as_ref() {
        return Some(seeds.clone());
    }
    cfg.frozen.as_ref().map(|f| {
        f.iter()
            .enumerate()
            .filter(|(_, on)| !**on)
            .map(|(i, _)| i)
            .collect()
    })
}

#[cfg(feature = "bank-rpc")]
fn pack_merge() -> f64 {
    #[cfg(feature = "featomic")]
    {
        crate::featomic_hop::SOAP_PACK_MERGE
    }
    #[cfg(not(feature = "featomic"))]
    {
        0.10
    }
}

#[cfg(feature = "bank-rpc")]
fn packing_of(x: ArrayView1<f64>, cfg: &Config) -> Array1<f64> {
    #[cfg(feature = "featomic")]
    {
        let mobile = mobile_of(cfg);
        crate::featomic_hop::soap_cloud_mean(
            x,
            3.5 * cfg.length_scale,
            cfg.species.as_deref(),
            mobile.as_deref(),
        )
    }
    #[cfg(not(feature = "featomic"))]
    {
        let _ = (x, cfg);
        Array1::zeros(0)
    }
}

/// Good-Turing missing mass on shared packings. Saturated: enough
/// observations and few singletons, so the next start should leave.
#[cfg(feature = "bank-rpc")]
pub fn catalog_saturated(wells: &[(Array1<f64>, f64)], w0: f64) -> bool {
    let w0 = w0.max(1e-9);
    let mut n = 0u32;
    let mut n1 = 0u32;
    for (_, h) in wells {
        let v = (*h / w0).round().max(0.0) as u32;
        if v == 0 {
            continue;
        }
        n += v;
        if v == 1 {
            n1 += 1;
        }
    }
    n >= 12 && (n1 as f64 / n as f64) < 0.20
}

#[cfg(feature = "bank-rpc")]
fn packing_is_known(x: ArrayView1<f64>, cfg: &Config, wells: &[Array1<f64>]) -> bool {
    if wells.is_empty() {
        return false;
    }
    let s = packing_of(x, cfg);
    let merge = pack_merge();
    wells.iter().any(|w| {
        if w.len() != s.len() || s.is_empty() {
            return false;
        }
        s.iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            <= merge
    })
}

/// Recompute a bank member on the receiving engine before it can affect the walk.
#[cfg(feature = "bank-rpc")]
fn validate_bank_sample<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    charged_validations: &mut usize,
    reported_energy: f64,
    state: ArrayView1<f64>,
) -> Option<f64>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    if !reported_energy.is_finite() || state.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if !ledger.charge() {
        return None;
    }
    *charged_validations += 1;
    let (fresh_energy, gradient) = objective.value_and_gradient(state);
    if !fresh_energy.is_finite()
        || gradient.len() != state.len()
        || gradient.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let gradient_max = gradient
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if gradient_max >= cfg.record_gradient {
        return None;
    }
    let energy_tolerance = 1.0e-6 + 1.0e-10 * reported_energy.abs().max(fresh_energy.abs());
    if (reported_energy - fresh_energy).abs() > energy_tolerance {
        return None;
    }
    Some(fresh_energy)
}

/// Move the current packing mean into a hole of the shared SOAP cloud.
#[cfg(feature = "bank-rpc")]
fn leave_known_packing(
    x: ArrayView1<f64>,
    cfg: &Config,
    wells: &[Array1<f64>],
    ledger: &mut Ledger,
    relax: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    rng: &mut impl rand::Rng,
) -> Array1<f64> {
    #[cfg(feature = "featomic")]
    {
        let rcut = 3.5 * cfg.length_scale;
        let mobile = mobile_of(cfg);
        let mut y = crate::featomic_hop::step_into_hole(
            x,
            wells,
            crate::featomic_hop::SOAP_PACK_MERGE,
            rcut,
            cfg.species.as_deref(),
            mobile.as_deref(),
            rng,
        );
        pin_adsorbate_above_slab(&mut y, cfg);
        if ledger.remaining() < 8 {
            return y;
        }
        let steps = cfg.relax_steps.min(ledger.remaining());
        let (_e, mut q) = relax(ledger, y.view(), steps);
        pin_adsorbate_above_slab(&mut q, cfg);
        if !packing_is_known(q.view(), cfg, wells) && structure_is_sane(q.view(), sane_sep(cfg)) {
            return q;
        }
        let mut y2 = crate::featomic_hop::step_into_hole(
            q.view(),
            wells,
            crate::featomic_hop::SOAP_PACK_MERGE * 1.5,
            rcut,
            cfg.species.as_deref(),
            mobile.as_deref(),
            rng,
        );
        pin_adsorbate_above_slab(&mut y2, cfg);
        y2
    }
    #[cfg(not(feature = "featomic"))]
    {
        let _ = (cfg, wells, ledger, relax, rng);
        x.to_owned()
    }
}

/// One HQ chain against the Cap'n Proto bank: own walk, pull a win or
/// a new packing, leave when the shared catalog is saturated.
///
/// The start is the caller's geometry (a packed water cluster, a slab
/// plus adsorbate), not a random LJ sphere.
#[cfg(feature = "bank-rpc")]
pub fn search_from_bank<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    seed: u64,
    sock: &str,
) -> Result<(Outcome, RelaxStats), crate::methods::cluster_hopping::SearchError>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let _caller_stop = ledger.bind_stop();
    use crate::bank_rpc::BankClient;
    use crate::bias::BasinBias;
    use crate::diversity::DiversityAnnealer;
    use crate::methods::cluster_hopping::{ClusterFingerprint, run_with_bias};
    use rand::Rng;

    let mut stats = RelaxStats::default();
    let mut bank_validation_charged = 0usize;
    let paid = PaidCheck::default();
    let mut relax = warm_relax(objective, cfg, &mut stats, seed, &paid);
    let mut grad = gated_grad(objective, &paid);

    let mut cfg = cfg.clone();
    cfg.budget_window = true;
    let cfg = &cfg;
    let mut client = match BankClient::connect(sock) {
        Ok(c) => {
            println!("  capnp bank {sock} (informer; walk continues if it drops)");
            Some(c)
        }
        Err(e) => {
            println!("  bank {sock} down ({e}); own walk");
            return search_from(objective, cfg, ledger, start, seed);
        }
    };
    let sync_every = std::env::var("BANK_SYNC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    // Adopt the bank's deepest minimum when it beats this chain. The
    // default pull is an informer and stays quiet while the chain is
    // still descending in its own basin.
    let adopt_bank = std::env::var("BANK_ADOPT").ok().as_deref() == Some("1");
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(cfg, &start.to_owned()),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let slice = std::env::var("BANK_SLICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut hops = 0usize;
    let mut basins = 0usize;
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut slices = 0usize;
    let mut null_starts = 0usize;
    let (mut best, mut best_state, mut improvements) = empty_validated_record();
    let total = ledger.remaining();
    let mut schedule: Option<DiversityAnnealer> = None;
    let expected = 3 * cfg.n_points;

    let mut well_pairs: Vec<(Array1<f64>, f64)> = Vec::new();
    let mut catalog_best = f64::INFINITY;
    let mut catalog_size = 0usize;
    let mut stall = 0u32;
    while ledger.remaining() > 0 {
        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
        let pull = adopt_bank || slices == 0 || slices.is_multiple_of(sync_every);
        if pull && client.is_none() {
            client = BankClient::connect(sock).ok();
        }
        if pull && let Some(c) = client.as_mut() {
            match c.snapshot() {
                Ok(s) => {
                    for (soap, h) in &s.wells {
                        bias.import_well(soap.clone(), *h);
                    }
                    #[cfg(feature = "featomic")]
                    crate::featomic_hop::set_packing_archive(
                        s.wells.iter().map(|(soap, _)| soap.clone()).collect(),
                    );
                    well_pairs = s.wells.clone();
                    catalog_size = s.size as usize;
                    if !s.energies.is_empty() {
                        catalog_best = s.energies.iter().copied().fold(f64::INFINITY, f64::min);
                    }
                    if s.size >= 2 {
                        let sched = schedule.get_or_insert_with(|| {
                            DiversityAnnealer::from_initial(s.dcut.max(pack_merge()))
                                .with_final_fraction(0.4)
                        });
                        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
                        if c.set_dcut(sched.threshold(progress)).is_err() {
                            client = None;
                        }
                    }
                }
                Err(_) => client = None,
            }
        }
        let wells: Vec<Array1<f64>> = well_pairs.iter().map(|(w, _)| w.clone()).collect();
        let mut start = best_state.clone().unwrap_or_else(|| start.to_owned());
        let mine = packing_of(start.view(), cfg);
        let gap = {
            #[cfg(feature = "featomic")]
            {
                crate::featomic_hop::SOAP_PACK_GAP
            }
            #[cfg(not(feature = "featomic"))]
            {
                0.15
            }
        };
        let sat = catalog_saturated(&well_pairs, cfg.bias_height);
        let on_known = packing_is_known(start.view(), cfg, &wells);
        let swarm = crate::swarm::policy(
            progress,
            best,
            best,
            catalog_best,
            catalog_size,
            sat,
            on_known && sat,
            stall,
        );
        if adopt_bank
            && catalog_best < best - 1e-6
            && let Some(c) = client.as_mut()
        {
            match c.sample(u64::MAX) {
                Ok(Some((reported_energy, x)))
                    if x.len() == expected
                        && reported_energy < best - 1e-6
                        && structure_is_sane(x.view(), sane_sep(cfg)) =>
                {
                    let fresh_energy = validate_bank_sample(
                        objective,
                        cfg,
                        ledger,
                        &mut bank_validation_charged,
                        reported_energy,
                        x.view(),
                    );
                    if let Some(energy) = fresh_energy.filter(|energy| *energy < best - 1e-6) {
                        println!("      bank adopt {energy:.6} over {best:.6} at slice {slices}");
                        best = energy;
                        best_state = Some(x.clone());
                        if improvements.len() < 512 {
                            improvements.push((hops, ledger.spent(), bias.n_basins(), energy));
                        }
                        start = x;
                    }
                }
                Ok(_) => {}
                Err(_) => client = None,
            }
        }
        if pull
            && swarm.pull
            && let Some(c) = client.as_mut()
        {
            match c.sample(rng.random()) {
                Ok(Some((reported_energy, x)))
                    if x.len() == expected && structure_is_sane(x.view(), sane_sep(cfg)) =>
                {
                    let theirs = packing_of(x.view(), cfg);
                    let dist = if mine.is_empty() || mine.len() != theirs.len() {
                        f64::INFINITY
                    } else {
                        mine.iter()
                            .zip(theirs.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f64>()
                            .sqrt()
                    };
                    let worth_validating = if swarm.win_only {
                        reported_energy < best - 0.05
                    } else {
                        reported_energy < best - 0.05
                            || (dist > gap && reported_energy < best + 1.0)
                    };
                    let fresh_energy = worth_validating
                        .then(|| {
                            validate_bank_sample(
                                objective,
                                cfg,
                                ledger,
                                &mut bank_validation_charged,
                                reported_energy,
                                x.view(),
                            )
                        })
                        .flatten();
                    let take = fresh_energy.is_some_and(|energy| {
                        if swarm.win_only {
                            energy < best - 0.05
                        } else {
                            energy < best - 0.05 || (dist > gap && energy < best + 1.0)
                        }
                    });
                    if take {
                        let energy = fresh_energy.expect("take requires fresh bank evidence");
                        if energy < best {
                            best = energy;
                            best_state = Some(x.clone());
                            if improvements.len() < 512 {
                                improvements.push((hops, ledger.spent(), bias.n_basins(), energy));
                            }
                        }
                        start = x;
                    }
                }
                Ok(_) => {}
                Err(_) => client = None,
            }
        }
        if ledger.remaining() == 0 {
            break;
        }
        if swarm.leave {
            null_starts += 1;
            start = leave_known_packing(start.view(), cfg, &wells, ledger, &mut relax, &mut rng);
        }
        let charged_before = ledger.spent();
        let hops_before = hops;
        let mut slice_led = ledger.slice(slice.min(ledger.remaining()));
        let out = run_with_bias(
            cfg,
            start.view(),
            &mut slice_led,
            &mut relax,
            Some(&mut grad),
            &mut bias,
            &mut rng,
        )?;
        ledger.charge_many(slice_led.spent());
        if let Some(st) = slice_led.best_state.as_ref() {
            ledger.record(slice_led.best, st.view());
        }
        hops += out.hops;
        basins += out.basins;
        screened_out += out.screened_out;
        returned += out.returned;
        slices += 1;
        for (h, c, b, e) in out.improvements {
            if improvements.len() >= 512 {
                break;
            }
            improvements.push((h + hops_before, c + charged_before, b, e));
        }
        let improved = matches!(out.best_state.as_ref(), Some(st) if out.best < best && structure_is_sane(st.view(), sane_sep(cfg)));
        if improved {
            stall = 0;
        } else {
            stall = stall.saturating_add(1);
        }
        if improved {
            best = out.best;
            best_state = out.best_state.clone();
            if let Some(st) = out.best_state.as_ref()
                && let Some(c) = client.as_mut()
            {
                let soap = packing_of(st.view(), cfg);
                if c.offer(out.best, st.view(), soap.view()).is_err()
                    || c.deposit(soap.view(), cfg.bias_height).is_err()
                {
                    client = None;
                }
            }
        }
    }
    println!(
        "      capnp bank: {slices} slices, {null_starts} archive-null starts, best {best:.6}"
    );
    drop(relax);
    stats.check_charged += bank_validation_charged;
    let out = Outcome {
        best,
        best_state,
        hops,
        basins,
        screened_out,
        returned,
        charged: ledger.spent(),
        improvements,
        ..Outcome::default()
    };
    Ok((out, stats))
}

#[cfg(any(feature = "bank-rpc", test))]
type ImprovementRecord = (usize, usize, usize, f64);

#[cfg(any(feature = "bank-rpc", test))]
fn empty_validated_record() -> (f64, Option<Array1<f64>>, Vec<ImprovementRecord>) {
    (f64::INFINITY, None, Vec::new())
}

/// Work spent before a run first reached `target`, or how much it spent
/// without reaching it.
///
/// The statistic to report. A success rate at a fixed budget is this quantity
/// pushed through an arbitrary threshold: above the budget it saturates and
/// hides the margin, below it censors and hides how near the failures came.
/// Eight seeds in eight at twelve million evaluations and five in eight at
/// three million are the same method described twice, badly.
///
/// A first encounter time is a property of the method. It is what lets one
/// paper's result be compared with another's, and it is what makes a claim like
/// a seventyfold improvement mean something.
///
/// # Censoring
///
/// A run that never reached the target has not produced a first encounter time;
/// it has produced a lower bound. That is [`Encounter::Censored`], and it must
/// not be dropped or replaced by the budget: dropping the failures reports the
/// mean of the successes, which is smaller than the truth and gets smaller as
/// the method gets worse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Encounter {
    /// Charged evaluations spent when the target was first reached.
    Found {
        /// Charged evaluations at the first crossing.
        charged: usize,
        /// Hops at the first crossing.
        hops: usize,
    },
    /// The target was never reached; the run spent this much without it.
    Censored {
        /// Charged evaluations spent in total.
        charged: usize,
    },
}

impl Encounter {
    /// The charged count either way, which is the encounter time when found and
    /// a lower bound on it when censored.
    pub fn charged(&self) -> usize {
        match self {
            Encounter::Found { charged, .. } | Encounter::Censored { charged } => *charged,
        }
    }

    /// Whether the target was reached.
    pub fn found(&self) -> bool {
        matches!(self, Encounter::Found { .. })
    }
}

/// The first encounter with `target` in a run's improvement trace.
///
/// `target` is compared with a tolerance, since a published minimum is quoted
/// to six decimals and a relaxation lands near it rather than on it.
pub fn first_encounter(out: &Outcome, target: f64, tolerance: f64, spent: usize) -> Encounter {
    for &(hops, charged, _, e) in &out.improvements {
        if e < target + tolerance {
            return Encounter::Found { charged, hops };
        }
    }
    Encounter::Censored { charged: spent }
}

/// Median first encounter time under censoring, by Kaplan-Meier.
///
/// The median is the point where the survival function first falls to a half.
/// `None` when more than half the runs are censored, which is the honest answer:
/// the median has not been observed, and quoting the mean of the successes
/// instead reports a number that improves as the method gets worse.
///
/// The survival is kept as an exact ratio of products of at-risk counts. It is
/// exactly a half whenever half the runs are found before any is censored, and
/// a floating-point product of the same factors can round to either side of a
/// half, which moves the median by a whole encounter. A run censored at the
/// time of an encounter is still at risk at it, as Kaplan and Meier count it,
/// so the median does not depend on the order of `runs`.
pub fn median_encounter(runs: &[Encounter]) -> Option<usize> {
    let mut events: Vec<(usize, bool)> = runs.iter().map(|e| (e.charged(), e.found())).collect();
    events.sort_by_key(|&(c, found)| (c, !found));

    // survival = left / entered, the products of the at-risk counts after and
    // before each encounter, so it is at most a half when 2 left <= entered.
    let mut twice_left = Natural(vec![2]);
    let mut entered = Natural(vec![1]);
    let mut at_risk = events.len();
    for (c, found) in events {
        if found {
            twice_left.scale(at_risk - 1);
            entered.scale(at_risk);
            if twice_left <= entered {
                return Some(c);
            }
        }
        at_risk -= 1;
    }
    None
}

/// A natural number in base-2^32 limbs, least significant first, with no high
/// zero limb above the first.
#[derive(PartialEq, Eq)]
struct Natural(Vec<u32>);

impl Natural {
    fn scale(&mut self, k: usize) {
        let mut carry = 0_u128;
        for limb in &mut self.0 {
            let v = u128::from(*limb) * k as u128 + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        while carry > 0 {
            self.0.push(carry as u32);
            carry >>= 32;
        }
        while self.0.len() > 1 && self.0.last() == Some(&0) {
            self.0.pop();
        }
    }
}

impl Ord for Natural {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Checks that a reported result is what it claims to be.
///
/// Returns the energy of the returned structure and its largest gradient
/// component, both computed off the ledger and outside the driver. `None` when
/// no structure came back at all.
///
/// Worth having as a function rather than as a line in each example, because
/// checking only the energy is not enough: an arm of this crate once returned a
/// structure carrying the right energy with a gradient of 0.31, which is not a
/// minimum, and the energy check passed.
pub fn verify<O>(objective: &O, out: &Outcome) -> Option<(f64, f64)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let x = out.best_state.as_ref()?;
    let (e, g) = objective.value_and_gradient(x.view());
    let gmax = if g.iter().any(|value| !value.is_finite()) {
        f64::INFINITY
    } else {
        g.iter()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()))
    };
    Some((e, gmax))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::PairPotential;
    use ndarray::Array1;

    #[test]
    fn convergence_uses_the_configured_recording_threshold() {
        let gradient = Array1::from_vec(vec![7.2e-5, -3.0e-5]);
        assert!(gradient_is_converged(gradient.view(), 1.0e-3));
        assert!(!gradient_is_converged(gradient.view(), 1.0e-5));
    }

    /// An all-NaN gradient is a failed evaluation. `f64::max` drops NaN, so
    /// the infinity-norm fold used to report that failure as a zero gradient.
    #[test]
    fn a_nonfinite_gradient_is_not_converged() {
        let broken = Array1::from_vec(vec![f64::NAN, f64::NAN]);
        assert!(
            !gradient_is_converged(broken.view(), 1.0e-3),
            "NaN gradient was scored as converged"
        );
        let empty = Array1::from_vec(vec![]);
        assert!(!gradient_is_converged(empty.view(), 1.0e-3));
    }

    /// A non-finite evaluation must not become the recorded best, and a
    /// finite quench that the line search can still reach must.
    ///
    /// The first steepest step from the start leaves the finite bowl and
    /// the objective answers −∞ with a zero gradient. Adopting that answer
    /// ends the quench there: the best stays +∞, and the finite minimum is
    /// never eligible. A shorter step stays inside the bowl.
    #[test]
    fn finite_quench_energy_is_eligible_and_nonfinite_is_not_best() {
        use eindir_core::gradient::Gradient;
        use eindir_core::{Bounds, Objective};

        let state = Array1::zeros(1);
        let mut ledger = Ledger::new(1);
        ledger.record(f64::NAN, state.view());
        ledger.record(f64::INFINITY, state.view());
        ledger.record(f64::NEG_INFINITY, state.view());
        assert!(
            ledger.best.is_infinite() && ledger.best.is_sign_positive(),
            "non-finite eval became best {}",
            ledger.best
        );
        assert!(ledger.best_state.is_none());
        ledger.record(-2.5, state.view());
        assert_eq!(ledger.best, -2.5);
        ledger.record(f64::NEG_INFINITY, state.view());
        ledger.record(f64::NAN, state.view());
        assert_eq!(ledger.best, -2.5);

        struct Bowl {
            bounds: Bounds<f64>,
            target: f64,
            stiffness: f64,
            lo: f64,
            hi: f64,
        }

        impl Bowl {
            fn pair(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                let place = x.get(3).copied().unwrap_or(f64::NAN);
                if !place.is_finite() || place < self.lo || place > self.hi {
                    return (f64::NEG_INFINITY, Array1::zeros(x.len()));
                }
                let mut energy = 0.0;
                let mut gradient = Array1::zeros(x.len());
                for (i, coordinate) in x.iter().enumerate() {
                    let aim = if i == 3 { self.target } else { 0.0 };
                    let delta = coordinate - aim;
                    energy += 0.5 * self.stiffness * delta * delta;
                    gradient[i] = self.stiffness * delta;
                }
                (energy, gradient)
            }
        }

        impl Objective<f64> for Bowl {
            fn dim(&self) -> usize {
                6
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.bounds
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                self.pair(x).0
            }
        }

        impl Gradient<f64> for Bowl {
            fn dim(&self) -> usize {
                6
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                self.pair(x).1
            }
        }

        impl DifferentiableObjective<f64> for Bowl {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                self.pair(x)
            }
        }

        let bowl = Bowl {
            bounds: Bounds::new(Array1::from_elem(6, -8.0), Array1::from_elem(6, 8.0), 0.0),
            target: 1.5,
            stiffness: 10.0,
            lo: 0.4,
            hi: 1.8,
        };
        let mut cfg = Config::for_cluster(2);
        cfg.max_hops = Some(0);
        let start = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut ledger = Ledger::new(2_000);
        let (out, _) = search_from(&bowl, &cfg, &mut ledger, start.view(), 1)
            .expect("the bowl has a finite quench");
        assert!(
            out.best.is_finite(),
            "finite quench was not eligible, best {}",
            out.best
        );
        assert!(
            out.best < 1.0e-4,
            "best {} is not the finite minimum of the bowl",
            out.best
        );
        assert!(out.best_state.is_some(), "finite best has no geometry");
        let best = out.best_state.expect("finite best");
        assert!(
            best.iter().all(|value| value.is_finite()),
            "best geometry is not finite"
        );
        assert!(
            (best[3] - bowl.target).abs() < 1.0e-3,
            "best landed at {} instead of the finite minimum",
            best[3]
        );
    }

    /// The water Good-Turing arm retires only from quench boundaries. A search
    /// that never records them cannot saturate, so this is the hook's input.
    #[test]
    fn production_quenches_reach_the_census_hook() {
        use crate::methods::cluster_hopping::{ChainCheckpoint, CheckpointAction};

        let potential = PairPotential::lennard_jones(5);
        let cfg = Config::for_cluster(5);
        let mut ledger = Ledger::new(180);
        let start = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.2, 0.2, 0.0, 0.2, 1.3, 0.1, -0.4, 0.5, 1.1, 0.7, -0.6, 0.8,
        ]);
        let mut observed = 0usize;
        let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
            observed += snapshot.quench_boundaries().len();
            CheckpointAction::Continue
        };
        search_from_with_checkpoint(
            &potential,
            &cfg,
            &mut ledger,
            start.view(),
            1,
            1,
            &mut checkpoint,
        )
        .expect("finite start");
        assert!(
            !ledger.quench_boundaries().is_empty(),
            "search recorded no quench boundary"
        );
        assert!(
            observed > 0,
            "checkpoint saw none of the recorded boundaries"
        );
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn bank_samples_require_charged_local_energy_and_gradient_validation() {
        let potential = PairPotential::lennard_jones(2);
        let config = Config::for_cluster(2);
        let radius = 2.0_f64.powf(1.0 / 6.0) / 2.0;
        let minimum = Array1::from_vec(vec![-radius, 0.0, 0.0, radius, 0.0, 0.0]);
        let mut ledger = Ledger::new(3);
        let mut charged_validations = 0usize;

        let accepted = validate_bank_sample(
            &potential,
            &config,
            &mut ledger,
            &mut charged_validations,
            -1.0,
            minimum.view(),
        );

        assert!((accepted.expect("validated minimum was rejected") + 1.0).abs() < 1e-12);
        assert_eq!(ledger.spent(), 1);
        assert_eq!(charged_validations, 1);

        let rejected = validate_bank_sample(
            &potential,
            &config,
            &mut ledger,
            &mut charged_validations,
            -0.5,
            minimum.view(),
        );
        assert!(rejected.is_none(), "a peer energy mismatch was adopted");
        assert_eq!(ledger.spent(), 2);
        assert_eq!(charged_validations, 2);

        let separated = 1.2_f64;
        let displaced =
            Array1::from_vec(vec![-separated / 2.0, 0.0, 0.0, separated / 2.0, 0.0, 0.0]);
        let inverse_sixth = separated.powi(-6);
        let displaced_energy = 4.0 * (inverse_sixth * inverse_sixth - inverse_sixth);
        let unquenched = validate_bank_sample(
            &potential,
            &config,
            &mut ledger,
            &mut charged_validations,
            displaced_energy,
            displaced.view(),
        );
        assert!(unquenched.is_none(), "a nonminimum bank sample was adopted");
        assert_eq!(ledger.spent(), 3);
        assert_eq!(charged_validations, 3);

        let unaffordable = validate_bank_sample(
            &potential,
            &config,
            &mut ledger,
            &mut charged_validations,
            -1.0,
            minimum.view(),
        );
        assert!(unaffordable.is_none());
        assert_eq!(ledger.spent(), 3);
        assert_eq!(charged_validations, 3);
    }

    #[test]
    fn bank_search_does_not_report_its_unquenched_start() {
        let (best, best_state, improvements) = empty_validated_record();
        assert!(best.is_infinite() && best.is_sign_positive());
        assert!(best_state.is_none());
        assert!(improvements.is_empty());
    }

    /// A caller that raises the ledger stop after N potential calls must get
    /// the search back at once, with the best state found and with no further
    /// calls. Basin hopping, the communicating preset, the recommended preset,
    /// a replica ladder, and the species chart all share this. The rigid
    /// TIP4P driver is covered with the stock searches.
    #[test]
    fn caller_stop_returns_at_once() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};

        use eindir_core::{Gradient, Objective};

        const AFTER: usize = 80;
        const BUDGET: usize = 80_000;

        struct Halt {
            inner: PairPotential,
            stop: Arc<AtomicBool>,
            calls: AtomicUsize,
            best: Mutex<f64>,
        }

        impl Objective<f64> for Halt {
            fn dim(&self) -> usize {
                Objective::dim(&self.inner)
            }
            fn bounds(&self) -> &eindir_core::Bounds<f64> {
                self.inner.bounds()
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                self.value_and_gradient(x).0
            }
        }

        impl Gradient<f64> for Halt {
            fn dim(&self) -> usize {
                Objective::dim(self)
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                self.value_and_gradient(x).1
            }
        }

        impl DifferentiableObjective<f64> for Halt {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                let (energy, gradient) = self.inner.value_and_gradient(x);
                let mut best = self.best.lock().expect("best");
                if energy < *best {
                    *best = energy;
                }
                if n + 1 >= AFTER {
                    self.stop.store(true, Ordering::SeqCst);
                }
                assert!(
                    n < AFTER + 8,
                    "the search kept calling the potential after the stop ({n} calls)"
                );
                (energy, gradient)
            }
        }

        let presets = [
            ("basin hopping", Config::for_cluster(8)),
            ("communicating", Config::communicating(8)),
            ("recommended", Config::recommended(8)),
            ("replica ladder", {
                let mut cfg = Config::for_cluster(8);
                cfg.replicas = 3;
                cfg.swap_period = 2;
                cfg
            }),
            (
                "species chart",
                Config::for_cluster(8).with_species(
                    vec![1, 1, 1, 1, 2, 2, 2, 2],
                    Some(vec![1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3]),
                ),
            ),
        ];
        for (name, cfg) in presets {
            let stop = Arc::new(AtomicBool::new(false));
            let objective = Halt {
                inner: PairPotential::lennard_jones(cfg.n_points),
                stop: Arc::clone(&stop),
                calls: AtomicUsize::new(0),
                best: Mutex::new(f64::INFINITY),
            };
            let mut ledger = Ledger::with_stop(BUDGET, Arc::clone(&stop));
            let started = Instant::now();
            let (out, _) = search(&objective, &cfg, &mut ledger, 1).expect("finite start");
            let elapsed = started.elapsed();
            let calls = objective.calls.load(Ordering::SeqCst);
            let best_seen = *objective.best.lock().expect("best");
            assert!(
                calls <= AFTER + 8,
                "{name} made {calls} calls after a stop at {AFTER}"
            );
            assert!(
                elapsed < Duration::from_secs(2),
                "{name} returned in {elapsed:?} with {calls} calls, {} hops, {} spent",
                out.hops,
                ledger.spent()
            );
            assert!(
                out.hops < 8,
                "{name} kept hopping ({}) after the stop",
                out.hops
            );
            if out.best.is_finite() {
                assert!(out.best_state.is_some(), "{name} dropped the best state");
                assert!(
                    out.best <= best_seen + 1e-8,
                    "{name} returned {} above the best energy seen {best_seen}",
                    out.best
                );
            }
        }
    }

    /// A stop flag that stays down must not change how many evaluations a
    /// search charges. The branch that reads the flag is taken on every
    /// charge, so a difference here is a change in the force-call count.
    #[test]
    fn an_idle_stop_flag_does_not_change_force_calls() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let pot = PairPotential::lennard_jones(6);
        let mut plain = Config::for_cluster(6);
        plain.relax_steps = 40;
        plain.screen_steps = 8;
        let mut ledger_plain = Ledger::new(800);
        let (out_plain, _) = search(&pot, &plain, &mut ledger_plain, 3).expect("finite start");
        let stop = Arc::new(AtomicBool::new(false));
        let mut ledger_flag = Ledger::with_stop(800, stop);
        let (out_flag, _) = search(&pot, &plain, &mut ledger_flag, 3).expect("finite start");
        assert_eq!(
            ledger_plain.spent(),
            ledger_flag.spent(),
            "spent {} with the flag down and {} without it",
            ledger_flag.spent(),
            ledger_plain.spent()
        );
        assert_eq!(out_plain.charged, out_flag.charged);
        assert_eq!(out_plain.hops, out_flag.hops);
        assert_eq!(out_plain.best, out_flag.best);
    }

    /// The search runs against a potential passed as a trait object, which is
    /// the whole point: an rgpot potential arrives the same way.
    #[test]
    fn it_searches_an_objective_given_as_a_trait_object() {
        let pot = PairPotential::lennard_jones(13);
        let dyn_pot: &dyn DifferentiableObjective<f64> = &pot;
        let mut cfg = Config::for_cluster(13);
        cfg.allocate_moves = true;
        cfg.return_screen = true;
        let mut ledger = Ledger::new(120_000);
        let (out, stats) = search(dyn_pot, &cfg, &mut ledger, 0).expect("finite start");

        assert!(out.hops > 0, "no hops were taken");
        assert!(stats.total() > 0, "no relaxations were counted");
        assert!(
            stats.converged > 0,
            "nothing converged, so the chain is not on the quenched landscape"
        );
        assert!(ledger.spent() <= 120_000, "spent {}", ledger.spent());
    }

    /// The returned structure has to be a minimum carrying the reported energy,
    /// and `verify` is what says so.
    #[test]
    fn verify_reports_the_energy_and_the_gradient_of_what_came_back() {
        let pot = PairPotential::lennard_jones(13);
        let mut cfg = Config::for_cluster(13);
        cfg.return_screen = true;
        let mut ledger = Ledger::new(120_000);
        let (out, _) = search(&pot, &cfg, &mut ledger, 3).expect("finite start");
        let (e, gmax) = verify(&pot, &out).expect("no structure came back");
        assert!(
            (e - out.best).abs() < 1e-6,
            "reported {} but the structure is {e}",
            out.best
        );
        assert!(gmax < 1e-3, "returned a structure with gradient {gmax:.2e}");
    }

    /// A run that reached the target reports where, and one that did not is
    /// censored rather than being given the budget as its time.
    #[test]
    fn an_encounter_is_found_or_censored() {
        let pot = PairPotential::lennard_jones(13);
        let mut cfg = Config::for_cluster(13);
        cfg.allocate_moves = true;
        cfg.return_screen = true;
        let mut ledger = Ledger::new(150_000);
        let (out, _) = search(&pot, &cfg, &mut ledger, 0).expect("finite start");
        let e = first_encounter(&out, -44.326801, 1e-4, ledger.spent());
        match e {
            Encounter::Found { charged, hops } => {
                assert!(charged > 0 && charged <= ledger.spent());
                assert!(hops > 0 && hops <= out.hops);
            }
            Encounter::Censored { charged } => assert_eq!(charged, ledger.spent()),
        }

        // A target nothing can reach must censor at the spend, not report a
        // time.
        let never = first_encounter(&out, -1e9, 1e-4, ledger.spent());
        assert_eq!(
            never,
            Encounter::Censored {
                charged: ledger.spent()
            }
        );
    }

    /// The median under censoring, and the refusal that keeps it honest.
    #[test]
    fn the_median_refuses_when_most_runs_are_censored() {
        let found = |c: usize| Encounter::Found {
            charged: c,
            hops: c / 30,
        };
        let cens = |c: usize| Encounter::Censored { charged: c };

        // Five found, spread; the median is the third.
        let all = vec![found(10), found(20), found(30), found(40), found(50)];
        assert_eq!(median_encounter(&all), Some(30));

        // One found early, four censored late: the survival function never
        // reaches a half, so there is no median to quote.
        let mostly = vec![found(10), cens(90), cens(91), cens(92), cens(93)];
        assert_eq!(median_encounter(&mostly), None);

        // Censoring must not be treated as a success: replacing the censored
        // runs with successes at the same times gives a median, which is
        // exactly the error this guards against.
        let wrong = vec![found(10), found(90), found(91), found(92), found(93)];
        assert!(median_encounter(&wrong).is_some());
    }

    /// A censored run late in the ordering must not shrink the median.
    #[test]
    fn late_censoring_does_not_flatter_the_median() {
        let found = |c: usize| Encounter::Found {
            charged: c,
            hops: 1,
        };
        let cens = |c: usize| Encounter::Censored { charged: c };
        let clean = vec![found(10), found(20), found(30), found(40)];
        let with_censor = vec![found(10), found(20), found(30), cens(1000)];
        let a = median_encounter(&clean).unwrap();
        let b = median_encounter(&with_censor).unwrap();
        assert!(b >= a, "censoring moved the median from {a} down to {b}");
    }

    /// The survival as a floating-point product over the at-risk counts of
    /// successive encounters.
    fn float_survival(at_risk: impl Iterator<Item = usize>) -> f64 {
        at_risk.fold(1.0, |s, r| s * (1.0 - 1.0 / r as f64))
    }

    /// Half the runs found before any is censored puts the survival on a half
    /// exactly, so the median is the last of them. At these counts the
    /// floating-point product rounds above a half, and at 196 the exact
    /// products run past one limb.
    #[test]
    fn the_median_is_the_middle_encounter_at_even_counts() {
        for n in [24_usize, 28, 30, 38, 196] {
            let float = float_survival((n / 2 + 1..=n).rev());
            assert!(float > 0.5, "float survival {float} at {n} runs");
            let runs: Vec<Encounter> = (1..=n)
                .map(|c| Encounter::Found {
                    charged: c,
                    hops: c,
                })
                .collect();
            assert_eq!(median_encounter(&runs), Some(n / 2), "{n} runs");
        }
    }

    /// Censoring before the total can put the survival on a half too: two of
    /// thirty censored first leave 28 at risk, and five of fifteen found, two
    /// censored and two more found leave 2/3 * 7/8 * 6/7. The floating-point
    /// product rounds above a half in both.
    #[test]
    fn censoring_before_the_total_keeps_the_median_exact() {
        let found = |c: usize| Encounter::Found {
            charged: c,
            hops: 1,
        };
        let cens = |c: usize| Encounter::Censored { charged: c };

        assert!(float_survival((15..=28).rev()) > 0.5);
        let mut early = vec![cens(1), cens(2)];
        early.extend((10..38).map(found));
        assert_eq!(median_encounter(&early), Some(23));

        assert!(float_survival([15, 14, 13, 12, 11, 8, 7].into_iter()) > 0.5);
        let mut between: Vec<Encounter> = (1..=5).map(found).collect();
        between.extend([cens(6), cens(7)]);
        between.extend((8..=15).map(found));
        assert_eq!(median_encounter(&between), Some(9));
    }

    /// A run censored at the time of an encounter is still at risk at it, in
    /// whatever order the runs come: of three, one found and one censored at
    /// 5 leave the survival at 2/3, so the median is the encounter at 9.
    #[test]
    fn a_run_censored_at_an_encounter_is_still_at_risk() {
        let found = |c: usize| Encounter::Found {
            charged: c,
            hops: 1,
        };
        let cens = |c: usize| Encounter::Censored { charged: c };
        assert_eq!(median_encounter(&[cens(5), found(5), found(9)]), Some(9));
        assert_eq!(median_encounter(&[found(5), cens(5), found(9)]), Some(9));
    }

    /// LJ13 is the case with one answer everyone agrees on, so it is the one
    /// that says the plumbing did not quietly change the problem.
    #[test]
    fn it_finds_the_thirteen_point_icosahedron() {
        let pot = PairPotential::lennard_jones(13);
        let mut cfg = Config::for_cluster(13);
        cfg.allocate_moves = true;
        cfg.return_screen = true;
        let mut best = f64::INFINITY;
        for seed in 0..3 {
            let mut ledger = Ledger::new(150_000);
            let (out, _) = search(&pot, &cfg, &mut ledger, seed).expect("finite start");
            best = best.min(out.best);
        }
        assert!(
            best < -44.326801 + 1e-4,
            "best {best} against the published -44.326801"
        );
    }

    /// The budget is the experiment, so nothing may run past it.
    #[test]
    fn it_stops_at_the_budget() {
        let pot = PairPotential::morse(13, 6.0);
        let cfg = Config::for_cluster(13);
        let mut ledger = Ledger::new(5_000);
        let (_, _) = search(&pot, &cfg, &mut ledger, 1).expect("finite start");
        assert!(ledger.spent() <= 5_000, "spent {}", ledger.spent());
    }

    #[test]
    fn overlapping_atoms_are_not_sane() {
        let mut x = Array1::zeros(6);
        x[3] = 0.01;
        assert!(!structure_is_sane(x.view(), OVERLAP_SEPARATION));
        assert!(min_pair_distance(x.view()) < OVERLAP_SEPARATION);
    }

    #[test]
    fn a_separated_pair_is_sane() {
        let mut x = Array1::zeros(6);
        x[3] = 1.0;
        assert!(structure_is_sane(x.view(), OVERLAP_SEPARATION));
        assert!((min_pair_distance(x.view()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_covalent_hydrogen_pair_is_sane() {
        let mut x = Array1::zeros(6);
        x[3] = 0.74;
        assert!(
            structure_is_sane(x.view(), OVERLAP_SEPARATION),
            "H-H 0.74 Å must clear the overlap floor {OVERLAP_SEPARATION}",
        );
        const { assert!(OVERLAP_SEPARATION < 0.74) };
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn a_sparse_catalog_is_not_saturated() {
        let wells = vec![
            (Array1::from(vec![1.0]), 1.0),
            (Array1::from(vec![0.0]), 1.0),
        ];
        assert!(!catalog_saturated(&wells, 1.0));
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn a_full_catalog_with_few_singletons_is_saturated() {
        let mut wells = Vec::new();
        for i in 0..10 {
            wells.push((Array1::from(vec![i as f64]), 2.0));
        }
        wells.push((Array1::from(vec![99.0]), 1.0));
        // 10*2 + 1 = 21 observations, one singleton: n1/N = 1/21 < 0.2
        assert!(catalog_saturated(&wells, 1.0));
    }

    /// A carbon chain held by stiff bond, middling angle and soft torsion
    /// springs: the spread of curvatures a covalent molecule has.
    struct SpringChain {
        atoms: usize,
        bounds: eindir_core::Bounds<f64>,
    }

    impl SpringChain {
        fn new(atoms: usize) -> Self {
            Self {
                atoms,
                bounds: eindir_core::Bounds::new(
                    Array1::from_elem(3 * atoms, -50.0),
                    Array1::from_elem(3 * atoms, 50.0),
                    0.0,
                ),
            }
        }

        fn springs(&self) -> impl Iterator<Item = (usize, usize, f64, f64)> + '_ {
            let bonds = (0..self.atoms - 1).map(|i| (i, i + 1, 30.0, 1.52));
            let angles = (0..self.atoms - 2).map(|i| (i, i + 2, 5.0, 2.5));
            let torsions = (0..self.atoms - 3).map(|i| (i, i + 3, 0.5, 3.4));
            bonds.chain(angles).chain(torsions)
        }

        fn start(&self) -> Array1<f64> {
            Array1::from_iter((0..self.atoms).flat_map(|i| {
                let t = i as f64;
                [1.3 * t, 0.9 * (1.7 * t).sin(), 0.7 * (0.9 * t).cos()]
            }))
        }
    }

    impl eindir_core::Objective<f64> for SpringChain {
        fn dim(&self) -> usize {
            3 * self.atoms
        }

        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.value_and_gradient(x).0
        }
    }

    impl eindir_core::gradient::Gradient<f64> for SpringChain {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.value_and_gradient(x).1
        }

        fn dim(&self) -> usize {
            3 * self.atoms
        }
    }

    impl DifferentiableObjective<f64> for SpringChain {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let mut e = 0.0;
            let mut g = Array1::zeros(x.len());
            for (i, j, k, r0) in self.springs() {
                let d: Vec<f64> = (0..3).map(|c| x[3 * i + c] - x[3 * j + c]).collect();
                let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                e += 0.5 * k * (r - r0).powi(2);
                for c in 0..3 {
                    let f = k * (r - r0) * d[c] / r;
                    g[3 * i + c] += f;
                    g[3 * j + c] -= f;
                }
            }
            (e, g)
        }
    }

    fn chain_relax_cost(precondition: bool) -> (usize, f64) {
        let chain = SpringChain::new(30);
        let mut cfg = Config::for_molecular(vec![6; 30], vec![(0..30).collect()], 1.0);
        cfg.record_gradient = 1e-4;
        cfg.precondition_bonds = precondition;
        let paid = PaidCheck::default();
        let mut stats = RelaxStats::default();
        let mut led = Ledger::new(20_000);
        let mut relax = warm_relax(&chain, &cfg, &mut stats, 0, &paid);
        let (_, x) = relax(&mut led, chain.start().view(), 5_000);
        (
            led.spent(),
            largest_component(chain.value_and_gradient(x.view()).1.view()),
        )
    }

    #[test]
    fn a_bond_preconditioned_quench_settles_a_stiff_chain_in_fewer_calls() {
        let (plain, plain_g) = chain_relax_cost(false);
        let (pre, pre_g) = chain_relax_cost(true);
        assert!(plain_g <= 1e-4 && pre_g <= 1e-4, "{plain_g} {pre_g}");
        assert!(
            2 * pre < plain,
            "preconditioned {pre} calls against plain {plain}"
        );
    }

    #[test]
    fn the_convergence_check_reuses_the_descents_last_evaluation() {
        let chain = SpringChain::new(12);
        let log = std::sync::Mutex::new(Vec::<Array1<f64>>::new());
        struct Logged<'a>(&'a SpringChain, &'a std::sync::Mutex<Vec<Array1<f64>>>);
        impl eindir_core::Objective<f64> for Logged<'_> {
            fn dim(&self) -> usize {
                eindir_core::Objective::dim(self.0)
            }
            fn bounds(&self) -> &eindir_core::Bounds<f64> {
                self.0.bounds()
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                self.value_and_gradient(x).0
            }
        }
        impl eindir_core::gradient::Gradient<f64> for Logged<'_> {
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                self.value_and_gradient(x).1
            }
            fn dim(&self) -> usize {
                eindir_core::Objective::dim(self.0)
            }
        }
        impl DifferentiableObjective<f64> for Logged<'_> {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                self.1.lock().unwrap().push(x.to_owned());
                self.0.value_and_gradient(x)
            }
        }
        let logged = Logged(&chain, &log);
        let mut cfg = Config::for_molecular(vec![6; 12], vec![(0..12).collect()], 1.0);
        cfg.record_gradient = 1e-3;
        let paid = PaidCheck::default();
        let mut stats = RelaxStats::default();
        let mut led = Ledger::new(20_000);
        let mut relax = warm_relax(&logged, &cfg, &mut stats, 0, &paid);
        let (_, x) = relax(&mut led, chain.start().view(), 5_000);
        drop(relax);
        let points = log.lock().unwrap();
        assert_eq!(led.spent(), points.len(), "every charge is one evaluation");
        let at_end = points.iter().filter(|p| **p == x).count();
        assert_eq!(at_end, 1, "the end point was evaluated {at_end} times");
        assert_eq!(stats.check_charged, 0);
    }

    #[test]
    fn a_record_polish_descends_past_the_quench_stop() {
        let chain = SpringChain::new(12);
        let mut cfg = Config::for_molecular(vec![6; 12], vec![(0..12).collect()], 1.0);
        cfg.record_gradient = 1e-1;
        cfg.polish_records = cfg.relax_steps + 300;
        let paid = PaidCheck::default();
        let mut stats = RelaxStats::default();
        let mut led = Ledger::new(20_000);
        let mut relax = warm_relax(&chain, &cfg, &mut stats, 0, &paid);
        let (quenched, x) = relax(&mut led, chain.start().view(), cfg.relax_steps);
        let (polished, y) = relax(&mut led, x.view(), cfg.polish_records);
        drop(relax);
        assert!(largest_component(chain.value_and_gradient(x.view()).1.view()) <= 1e-1);
        assert!(polished < quenched, "polish {polished} quench {quenched}");
        assert!(largest_component(chain.value_and_gradient(y.view()).1.view()) < 1e-3);
    }

    #[test]
    fn the_molecular_record_gate_sits_above_a_gfn_quench_floor() {
        let cfg = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
        let reduced = 1.0e-3 / cfg.length_scale;
        assert!(
            cfg.record_gradient > 10.0 * reduced,
            "gate {:.3e} is still the reduced-unit floor {reduced:.3e}",
            cfg.record_gradient
        );
        assert!((cfg.record_gradient - 5.0e-2).abs() < 1e-15);
    }
}
