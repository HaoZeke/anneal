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

use crate::methods::cluster_hopping::{Config, Ledger, Outcome, optimize_with_gradient};
use crate::methods::warm_lbfgs::WarmLbfgs;
use crate::quench::{QuenchPredictor, Verdict};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;

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
pub const CONVERGED_GRADIENT: f64 = 1e-5;

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
) -> (Outcome, RelaxStats)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut stats = RelaxStats::default();
    let mut opt = WarmLbfgs::default();

    // Split deliberately: the relaxation needs the optimizer mutably, the
    // gradient needs only the objective. Sharing the objective by reference is
    // what lets both closures exist at once.
    // The driver calls the screening pass with `screen_steps` and the full one
    // with `relax_steps`, so the iteration count identifies which is which.
    let screen_iters = cfg.screen_steps;
    let adaptive = cfg.adaptive_screen;
    let probe = cfg.probe_screen;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let before = led.spent();
        let screening = iters <= screen_iters;
        // The screening pass stops when its own trajectory says the question is
        // settled; the full relaxation runs to its tolerance, because there the
        // answer is the structure and not a verdict about it.
        let mut pred = QuenchPredictor::new();
        pred.warmup = cfg.quench_warmup;
        pred.confidence = cfg.quench_confidence;
        let mut early = false;
        let mut probe_at: Option<(usize, f64)> = None;
        let target = led.best;
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(objective.value_and_gradient(v))
            },
            |_, fv| {
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
            },
        );
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
        // A descent stopped on a verdict is known not to be converged, so
        // asking is spending an evaluation on an answer already in hand. The
        // check stays on every other path, where the answer is not known.
        if early {
            stats.capped += 1;
            return (f, xr);
        }
        // Charged like any other evaluation: asking whether a relaxation
        // converged is asking the potential a question, and a protocol that
        // counts evaluations has to count this one.
        let converged = if led.charge() {
            stats.check_charged += 1;
            let g = objective.grad(xr.view());
            g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < CONVERGED_GRADIENT
        } else {
            false
        };
        if converged {
            stats.converged += 1;
        } else {
            stats.capped += 1;
        }
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(objective.grad(x))
    };

    let out = optimize_with_gradient(cfg, ledger, &mut relax, Some(&mut grad), seed);
    (out, stats)
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
) -> (Outcome, RelaxStats)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut stats = RelaxStats::default();
    let mut opt = WarmLbfgs::default();
    let screen_iters = cfg.screen_steps;
    let adaptive = cfg.adaptive_screen;
    let probe = cfg.probe_screen;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let before = led.spent();
        let screening = iters <= screen_iters;
        let mut pred = QuenchPredictor::new();
        pred.warmup = cfg.quench_warmup;
        pred.confidence = cfg.quench_confidence;
        let mut early = false;
        let mut probe_at: Option<(usize, f64)> = None;
        let target = led.best;
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(objective.value_and_gradient(v))
            },
            |_, fv| {
                if screening && probe {
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
                if pred.verdict(target) != Verdict::Hopeless {
                    return true;
                }
                early = true;
                false
            },
        );
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
        let f = if early {
            pred.stopped_energy(target, f)
        } else {
            f
        };
        if early {
            stats.capped += 1;
            return (f, xr);
        }
        let converged = if led.charge() {
            stats.check_charged += 1;
            let g = objective.grad(xr.view());
            g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < CONVERGED_GRADIENT
        } else {
            false
        };
        if converged {
            stats.converged += 1;
        } else {
            stats.capped += 1;
        }
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(objective.grad(x))
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let out = crate::methods::cluster_hopping::run_with_gradient(
        cfg,
        start,
        ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    );
    (out, stats)
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
) -> (Outcome, RelaxStats)
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

/// Shortest pair distance in a 3N Cartesian state.
pub fn min_pair_distance(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < best {
                best = r2;
            }
        }
    }
    best.sqrt()
}

/// A bank member is usable only if every coordinate is finite and no
/// two atoms sit on top of each other.
///
/// EAM and xTB both reward overlap with a huge negative energy. Adopting
/// that as a win fills the bank with a catastrophe and every later chain
/// copies it.
pub fn structure_is_sane(x: ArrayView1<f64>, min_sep: f64) -> bool {
    x.iter().all(|v| v.is_finite()) && min_pair_distance(x) >= min_sep
}

/// Keep adsorbate atoms above the frozen slab after a SOAP hole step.
#[cfg(feature = "bank-rpc")]
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
pub const OVERLAP_SEPARATION: f64 = 0.35;

fn sane_sep(_cfg: &Config) -> f64 {
    OVERLAP_SEPARATION
}

/// Mobile atom indices: the active region, or the complement of `frozen`.
#[cfg(feature = "bank-rpc")]
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
) -> (Outcome, RelaxStats)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    use crate::bank_rpc::BankClient;
    use crate::bias::BasinBias;
    use crate::diversity::DiversityAnnealer;
    use crate::methods::cluster_hopping::{ClusterFingerprint, run_with_bias};
    use rand::Rng;

    let mut stats = RelaxStats::default();
    let mut opt = WarmLbfgs::default();
    let screen_iters = cfg.screen_steps;
    let adaptive = cfg.adaptive_screen;
    let probe = cfg.probe_screen;
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let before = led.spent();
        let screening = iters <= screen_iters;
        let mut pred = QuenchPredictor::new();
        pred.warmup = cfg.quench_warmup;
        pred.confidence = cfg.quench_confidence;
        let mut early = false;
        let mut probe_at: Option<(usize, f64)> = None;
        let target = led.best;
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(objective.value_and_gradient(v))
            },
            |_, fv| {
                if screening && probe {
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
                if pred.verdict(target) != Verdict::Hopeless {
                    return true;
                }
                early = true;
                false
            },
        );
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
        let f = if early {
            pred.stopped_energy(target, f)
        } else {
            f
        };
        if early {
            stats.capped += 1;
            return (f, xr);
        }
        let converged = if led.charge() {
            stats.check_charged += 1;
            let g = objective.grad(xr.view());
            g.iter().fold(0.0_f64, |a, v| a.max(v.abs())) < CONVERGED_GRADIENT
        } else {
            false
        };
        if converged {
            stats.converged += 1;
        } else {
            stats.capped += 1;
        }
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(objective.grad(x))
    };

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
    let mut improvements: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut best = f64::INFINITY;
    let mut best_state: Option<Array1<f64>> = Some(start.to_owned());
    if structure_is_sane(start, sane_sep(cfg)) && ledger.charge() {
        let (e0, _) = objective.value_and_gradient(start);
        if e0.is_finite() {
            best = e0;
            improvements.push((0, ledger.spent(), 0, e0));
        }
    }
    let total = ledger.remaining();
    let mut schedule: Option<DiversityAnnealer> = None;
    let expected = 3 * cfg.n_points;

    let mut well_pairs: Vec<(Array1<f64>, f64)> = Vec::new();
    let mut catalog_best = f64::INFINITY;
    let mut catalog_size = 0usize;
    let mut stall = 0u32;
    while ledger.remaining() > 0 {
        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
        let pull = slices == 0 || slices.is_multiple_of(sync_every);
        if pull && client.is_none() {
            client = BankClient::connect(sock).ok();
        }
        if pull {
            if let Some(c) = client.as_mut() {
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
        let swarm = crate::swarm::decide_with_stall(
            progress,
            best,
            best,
            catalog_best,
            catalog_size,
            sat,
            on_known && sat,
            stall,
        );
        if pull && swarm.pull {
            if let Some(c) = client.as_mut() {
                match c.sample(rng.random()) {
                    Ok(Some((e, x)))
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
                        let take = if swarm.win_only {
                            e < best - 0.05
                        } else {
                            e < best - 0.05 || (dist > gap && e < best + 1.0)
                        };
                        if take {
                            if e < best {
                                best = e;
                                best_state = Some(x.clone());
                                if improvements.len() < 512 {
                                    improvements.push((hops, ledger.spent(), bias.n_basins(), e));
                                }
                            }
                            start = x;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => client = None,
                }
            }
        }
        if swarm.leave {
            null_starts += 1;
            start = leave_known_packing(start.view(), cfg, &wells, ledger, &mut relax, &mut rng);
        }
        let charged_before = ledger.spent();
        let hops_before = hops;
        let mut slice_led = Ledger::new(slice.min(ledger.remaining()));
        let out = run_with_bias(
            cfg,
            start.view(),
            &mut slice_led,
            &mut relax,
            Some(&mut grad),
            &mut bias,
            &mut rng,
        );
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
            if let Some(st) = out.best_state.as_ref() {
                if let Some(c) = client.as_mut() {
                    let soap = packing_of(st.view(), cfg);
                    if c.offer(out.best, st.view(), soap.view()).is_err()
                        || c.deposit(soap.view(), cfg.bias_height).is_err()
                    {
                        client = None;
                    }
                }
            }
        }
    }
    println!(
        "      capnp bank: {slices} slices, {null_starts} archive-null starts, best {best:.6}"
    );
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
    (out, stats)
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
pub fn median_encounter(runs: &[Encounter]) -> Option<usize> {
    if runs.is_empty() {
        return None;
    }
    let mut events: Vec<(usize, bool)> = runs.iter().map(|e| (e.charged(), e.found())).collect();
    events.sort_by_key(|(c, _)| *c);

    let mut at_risk = events.len() as f64;
    let mut survival = 1.0_f64;
    for (c, found) in events {
        if found {
            survival *= 1.0 - 1.0 / at_risk;
            if survival <= 0.5 {
                return Some(c);
            }
        }
        at_risk -= 1.0;
        if at_risk <= 0.0 {
            break;
        }
    }
    None
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
    Some((e, g.iter().fold(0.0_f64, |a, v| a.max(v.abs()))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::PairPotential;
    use ndarray::Array1;

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
        let (out, stats) = search(dyn_pot, &cfg, &mut ledger, 0);

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
        let (out, _) = search(&pot, &cfg, &mut ledger, 3);
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
        let (out, _) = search(&pot, &cfg, &mut ledger, 0);
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
            let (out, _) = search(&pot, &cfg, &mut ledger, seed);
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
        let (_, _) = search(&pot, &cfg, &mut ledger, 1);
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
            "H-H 0.74 Å must clear the overlap floor {}",
            OVERLAP_SEPARATION
        );
        assert!(OVERLAP_SEPARATION < 0.74);
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
}
