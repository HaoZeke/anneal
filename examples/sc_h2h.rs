//! Arm 3 on a Sutton-Chen cluster from the head-to-head's shared start.
//!
//! `iso` is one chain. `ens` is `replicas` chains that share one exact
//! minimum history and split the budget, with no shared bias, gossip or
//! restart. The chains take turns at each checkpoint, so a run is one
//! thread and repeats exactly. The Cambridge energy is a stopping threshold
//! only. Coordinates of the published minimum are never read.
//!
//! ```text
//! sc_h2h <iso|ens> <12-6|9-6|10-8> <n> <budget> <seed> <start-file> <reference> [replicas] [turn-calls]
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use anneal_core::descriptor_space::DescriptorVector;
use anneal_core::methods::cluster_hopping::{
    CheckpointAction, Config, Ledger, OrbitCore, Outcome, QuenchBoundary,
    run_with_history_at_checkpoints,
};
use anneal_core::methods::minima_hopping::{
    HistoryHook, HistoryMembership, HistoryReport, MinimumHistory, SerializedWitness,
    history_feedback_membership,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::pes_exploration::{ExactStructureWitness, StructureContext};
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use anneal_core::symmetrise::ShellCore;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Default basin-hopping quench of Wales's global minimisation program (GMIN):
/// root-mean-square (RMS) force `1e-3`, at most 500 steps.
const QUENCH_RMS: f64 = 1.0e-3;
const QUENCH_ITERS: usize = 500;
/// A hit is a validated quench at or below the Cambridge energy plus this window.
const HIT_WINDOW: f64 = 1.0e-3;
/// Two quenches are one minimum when their sorted pair distances differ by
/// less than this in Euclidean norm. The nearest neighbour is `1/sqrt(2)`.
const IDENTITY_RADIUS: f64 = 1.0e-3;

fn rms(g: &[f64]) -> f64 {
    let s: f64 = g.iter().map(|v| v * v).sum();
    (s / g.len().max(1) as f64).sqrt()
}

/// Sum of squared pair distances, `n` times the second moment about the centroid.
fn pair_square_sum(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    let mut s = 0.0;
    for i in 0..n {
        for k in 0..3 {
            let d = x[3 * i + k] - c[k] / n as f64;
            s += d * d;
        }
    }
    n as f64 * s
}

fn sorted_pairs(x: ArrayView1<f64>) -> Vec<f32> {
    let n = x.len() / 3;
    let mut d = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..3 {
                let v = x[3 * i + k] - x[3 * j + k];
                s += v * v;
            }
            d.push(s.sqrt() as f32);
        }
    }
    d.sort_by(f32::total_cmp);
    d
}

fn coordinate_key(x: ArrayView1<f64>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in x {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Sorted pair distances within [`IDENTITY_RADIUS`].
///
/// Every stored minimum is checked against each new one, so each structure's
/// sorted list is built once and kept by its coordinates. With `a` and `b`
/// the sorted lists, `|sum a^2 - sum b^2| <= |a - b| (|a| + |b|)`, so a gap in the
/// pair-square sums wider than the radius allows rejects before any lookup.
#[derive(Default)]
struct SortedPairsWitness {
    lists: Mutex<std::collections::HashMap<u64, Arc<Vec<f32>>>>,
}

impl SortedPairsWitness {
    fn list(&self, x: ArrayView1<f64>) -> Arc<Vec<f32>> {
        let key = coordinate_key(x);
        let mut lists = self.lists.lock().expect("pair lists");
        Arc::clone(
            lists
                .entry(key)
                .or_insert_with(|| Arc::new(sorted_pairs(x))),
        )
    }
}

impl ExactStructureWitness for SortedPairsWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        if left.len() != right.len() {
            return false;
        }
        let (sl, sr) = (pair_square_sum(left), pair_square_sum(right));
        if (sl - sr).abs() > IDENTITY_RADIUS * (sl.sqrt() + sr.sqrt()) {
            return false;
        }
        let (a, b) = (self.list(left), self.list(right));
        let bound = IDENTITY_RADIUS * IDENTITY_RADIUS;
        let mut s = 0.0;
        for (p, q) in a.iter().zip(b.iter()) {
            let d = f64::from(*p) - f64::from(*q);
            s += d * d;
            if s >= bound {
                return false;
            }
        }
        true
    }
}

/// Sorted distances from the centroid, which orders the witness checks.
fn radial_profile(x: ArrayView1<f64>) -> Vec<f64> {
    let n = x.len() / 3;
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k] / n as f64;
        }
    }
    let mut r: Vec<f64> = (0..n)
        .map(|i| {
            (0..3)
                .map(|k| (x[3 * i + k] - c[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    r.sort_by(f64::total_cmp);
    r
}

/// [`SharedMinimumHistory`]'s lock and membership rules with a cheap descriptor.
///
/// The library's universal descriptor costs about a fifth of a second per
/// quench at 78 points, more than the quench. The descriptor only orders the
/// witness checks, so the radial profile serves and identity is unchanged.
struct RadialHistory<'a, W: ExactStructureWitness + ?Sized> {
    history: &'a Mutex<MinimumHistory>,
    context: StructureContext,
    witness: &'a W,
    observations: usize,
    refusals: usize,
    seconds: f64,
}

impl<W: ExactStructureWitness + ?Sized> HistoryHook for RadialHistory<'_, W> {
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        let started = Instant::now();
        let report = (|| {
            let quench = QuenchBoundary::validated(energy, state.to_owned(), gradient.to_owned())?;
            let description = DescriptorVector::from_design(radial_profile(state)).ok()?;
            let mut history = self.history.lock().ok()?;
            let observation = history
                .observe(&quench, description, self.context.clone(), self.witness)
                .ok()?;
            let accepted = history.accepted_visits(observation.minimum.id)?;
            let (is_new, visits) = history_feedback_membership(
                HistoryMembership::Accepted,
                observation.minimum.is_new,
                observation.visits,
                accepted,
            );
            Some(HistoryReport {
                minimum: observation.minimum.id,
                is_new,
                visits,
                observed_visits: observation.visits,
                first_observation: observation.minimum.is_new,
            })
        })();
        self.seconds += started.elapsed().as_secs_f64();
        if report.is_some() {
            self.observations += 1;
        } else {
            self.refusals += 1;
        }
        report
    }

    fn mark_accepted(&mut self, minimum: usize) {
        let started = Instant::now();
        if let Ok(mut history) = self.history.lock() {
            let _ = history.mark_accepted(minimum);
        }
        self.seconds += started.elapsed().as_secs_f64();
    }

    fn cost(&self) -> (usize, usize, f64) {
        (self.observations, self.refusals, self.seconds)
    }
}

fn read_start(path: &str, n: usize) -> Array1<f64> {
    let text = std::fs::read_to_string(path).expect("start file");
    let coords: Vec<f64> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .take(3)
                .map(|v| v.parse::<f64>().expect("start coordinate"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(coords.len(), 3 * n, "{path} does not contain {n} points");
    Array1::from(coords)
}

/// Whose turn it is, which chains are still running, and the calls each has spent.
struct Turns {
    state: Mutex<(usize, Vec<bool>, Vec<usize>)>,
    changed: Condvar,
}

impl Turns {
    fn new(replicas: usize) -> Self {
        Self {
            state: Mutex::new((0, vec![true; replicas], vec![0; replicas])),
            changed: Condvar::new(),
        }
    }

    fn wait(&self, me: usize, stop: &AtomicBool) {
        let mut s = self.state.lock().expect("turns");
        while s.0 != me && !stop.load(Ordering::Relaxed) {
            s = self.changed.wait(s).expect("turns");
        }
    }

    /// Hands the turn on, records `spent`, and drops `me` from the rotation when done.
    fn pass(&self, me: usize, spent: usize, done: bool) {
        let mut s = self.state.lock().expect("turns");
        s.2[me] = spent;
        if done {
            s.1[me] = false;
        }
        let r = s.1.len();
        if let Some(next) = (1..=r).map(|k| (me + k) % r).find(|&q| s.1[q]) {
            s.0 = next;
        }
        self.changed.notify_all();
    }

    fn aggregate(&self, me: usize, mine: usize) -> usize {
        let s = self.state.lock().expect("turns");
        s.2.iter()
            .enumerate()
            .map(|(q, &v)| if q == me { mine } else { v })
            .sum()
    }
}

/// Replica index, seed, calls spent, outcome, and history cost.
type ReplicaEnd = (usize, u64, usize, Outcome, (usize, usize, f64));

struct Hit {
    calls: usize,
    quenches: usize,
    energy: f64,
    replica: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(8..=10).contains(&args.len()) {
        eprintln!(
            "usage: sc_h2h <iso|ens> <12-6|9-6|10-8> <n> <budget> <seed> <start-file> <reference> [replicas] [turn-calls]"
        );
        std::process::exit(2);
    }
    let arm = args[1].as_str();
    let family = SuttonChenFamily::parse(&args[2]).unwrap_or_else(|| {
        eprintln!("unknown family {}", args[2]);
        std::process::exit(2);
    });
    let n: usize = args[3].parse().expect("n");
    let budget: usize = args[4].parse().expect("budget");
    let seed: u64 = args[5].parse().expect("seed");
    let start = read_start(&args[6], n);
    let reference: f64 = args[7].parse().expect("reference energy");
    let replicas: usize = match arm {
        "iso" => 1,
        "ens" => args.get(8).map_or(4, |v| v.parse().expect("replicas")),
        _ => {
            eprintln!("unknown arm {arm}");
            std::process::exit(2);
        }
    };
    let turn_calls: usize = args.get(9).map_or(2000, |v| v.parse().expect("turn calls"));
    let target = reference + HIT_WINDOW;

    let potential = SuttonChen::new(n, family);
    let nn = std::f64::consts::FRAC_1_SQRT_2;
    let length_scale = nn / 2.0_f64.powf(1.0 / 6.0);
    let temperature = family.basin_hopping_temperature();
    // `with_scales` multiplies the Lennard-Jones temperature 0.8 by the energy
    // scale, so this scale makes the Metropolis temperature the published T*.
    let mut cfg = Config::with_scales(n, length_scale, temperature / 0.8);
    assert!((cfg.temperature - temperature).abs() < 1e-9);
    cfg.bayes_screen = true;
    cfg.relax_steps = QUENCH_ITERS;
    cfg.bayes_threshold =
        anneal_core::screen::cost_asymmetric_threshold(cfg.screen_steps, cfg.relax_steps);
    cfg.orbit_complete_on_new = true;
    cfg.orbit_core = OrbitCore::Both(ShellCore::scaled(cfg.length_scale));

    println!("code: anneal");
    println!("arm: {arm}");
    println!("replicas: {replicas}");
    println!("turn_calls: {turn_calls}");
    println!("family: {}", family.label());
    println!("n: {n}");
    println!("budget: {budget}");
    println!("seed: {seed}");
    println!("start: {}", args[6].rsplit('/').next().unwrap_or(""));
    println!("reference: {reference:.10}");
    println!("hit_window: {HIT_WINDOW}");
    println!("temperature: {temperature}");
    println!("length_scale: {length_scale:.10}");
    println!("record_gradient: {:.6e}", cfg.record_gradient);
    println!("identity_radius: {IDENTITY_RADIUS}");
    println!(
        "resolved_sha256: {}",
        cfg.resolved_sha256()
            .unwrap_or_else(|_| "unavailable".into())
    );

    let stop = Arc::new(AtomicBool::new(false));
    let full_quenches = AtomicUsize::new(0);
    let hit: Mutex<Option<Hit>> = Mutex::new(None);
    let turns = Turns::new(replicas);
    let history = Mutex::new(MinimumHistory::new(cfg.record_gradient).expect("history tolerance"));
    let context = StructureContext::new(
        Some(vec![79; n]),
        None,
        Some(format!("sutton-chen-{}-n{n}", family.label())),
    );
    let witness = SerializedWitness(Mutex::new(SortedPairsWitness::default()));
    let per_replica: Vec<usize> = (0..replicas)
        .map(|r| budget / replicas + usize::from(r < budget % replicas))
        .collect();

    let started = Instant::now();
    let outcomes: Vec<ReplicaEnd> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..replicas)
            .map(|r| {
                let cfg = &cfg;
                let stop = Arc::clone(&stop);
                let pot = potential.clone();
                let start = start.view();
                let (turns, hit, full_quenches) = (&turns, &hit, &full_quenches);
                let (history, context, witness) = (&history, &context, &witness);
                let budget = per_replica[r];
                let replica_seed =
                    seed.wrapping_add((r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                scope.spawn(move || {
                    turns.wait(r, &stop);
                    let mut ledger = Ledger::with_stop(budget, Arc::clone(&stop));
                    let mut rng = StdRng::seed_from_u64(replica_seed);
                    let mut opt = WarmLbfgs::default();
                    let screen_steps = cfg.screen_steps;
                    // The gradient at the end of a validated quench, so the
                    // record check at that point does not pay for it twice.
                    let last = Mutex::new(None::<(Array1<f64>, Array1<f64>)>);
                    let pot_r = pot.clone();
                    let last_r = &last;
                    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                        if iters == 0 || led.remaining() == 0 {
                            return if led.charge() {
                                (pot_r.value_and_gradient(x).0, x.to_owned())
                            } else {
                                (f64::INFINITY, x.to_owned())
                            };
                        }
                        let charged_before = led.spent();
                        opt.forget();
                        let full = iters > screen_steps;
                        // A 1e9 tolerance ends a screen at its first gradient check.
                        opt.gtol = if full { QUENCH_RMS } else { 1.0e9 };
                        let cap = if full { iters.min(QUENCH_ITERS) } else { iters };
                        let (mut energy, xr, _) = opt.minimize(x, cap, |v| {
                            led.charge().then(|| pot_r.value_and_gradient(v))
                        });
                        let mut grad = Array1::zeros(0);
                        if full && led.charge() {
                            let (e2, g2) = pot_r.value_and_gradient(xr.view());
                            energy = e2;
                            grad = g2;
                        }
                        let validated = full
                            && grad.len() == xr.len()
                            && rms(grad.as_slice().unwrap_or(&[])) <= QUENCH_RMS;
                        led.record_quench_boundary(
                            charged_before,
                            energy,
                            xr.clone(),
                            validated.then(|| grad.clone()),
                        );
                        if full {
                            let q = full_quenches.fetch_add(1, Ordering::Relaxed) + 1;
                            if validated && energy <= target {
                                let mut h = hit.lock().expect("hit");
                                if h.is_none() {
                                    led.record(energy, xr.view());
                                    *h = Some(Hit {
                                        calls: turns.aggregate(r, led.spent()).max(1),
                                        quenches: q,
                                        energy,
                                        replica: r,
                                    });
                                    stop.store(true, Ordering::Relaxed);
                                }
                            }
                            if grad.len() == xr.len() {
                                *last_r.lock().expect("last") = Some((xr.clone(), grad));
                            }
                        }
                        (energy, xr)
                    };
                    let pot_g = pot.clone();
                    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| {
                        if let Some((xs, g)) = last_r.lock().expect("last").as_ref()
                            && xs.view() == x
                        {
                            return Some(g.clone());
                        }
                        led.charge().then(|| pot_g.value_and_gradient(x).1)
                    };
                    let mut hook = RadialHistory {
                        history,
                        context: context.clone(),
                        witness,
                        observations: 0,
                        refusals: 0,
                        seconds: 0.0,
                    };
                    let hook_ref: Option<&mut dyn HistoryHook> =
                        (replicas > 1).then_some(&mut hook as &mut dyn HistoryHook);
                    let stop_c = Arc::clone(&stop);
                    let mut checkpoint =
                        |cp: anneal_core::methods::cluster_hopping::ChainCheckpoint<'_>| {
                            if replicas > 1 {
                                turns.pass(r, cp.charged(), false);
                                turns.wait(r, &stop_c);
                            }
                            CheckpointAction::Continue
                        };
                    let outcome = run_with_history_at_checkpoints(
                        cfg,
                        start,
                        &mut ledger,
                        &mut relax,
                        Some(&mut grad),
                        None,
                        hook_ref,
                        &mut rng,
                        turn_calls.max(1),
                        &mut checkpoint,
                    )
                    .unwrap_or_else(|err| {
                        eprintln!("replica {r} search error: {err:?}");
                        std::process::exit(1);
                    });
                    turns.pass(r, ledger.spent(), true);
                    let cost = hook.cost();
                    (r, replica_seed, ledger.spent(), outcome, cost)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("replica thread"))
            .collect()
    });
    let wall = started.elapsed().as_secs_f64();

    let mut spent = 0;
    let mut best = f64::INFINITY;
    for (r, rseed, charged, outcome, cost) in &outcomes {
        spent += charged;
        best = best.min(outcome.best);
        println!(
            "replica_{r}: seed={rseed} spent={charged} best={:.10} hops={} orbits={} restarts={} stall_escapes={} history_visits={}/{} history_cost={}/{}/{:.3}",
            outcome.best,
            outcome.hops,
            outcome.orbits.0,
            outcome.restarts,
            outcome.stall_escapes,
            outcome.history_visits.0,
            outcome.history_visits.1,
            cost.0,
            cost.1,
            cost.2,
        );
    }
    if replicas > 1 {
        let h = history.lock().expect("history");
        println!("history_minima: {}", h.minimum_count());
        println!("history_accepted: {}", h.accepted_count());
        println!("history_observations: {}", h.total_visits());
    }
    println!("wall_s: {wall:.6}");
    println!("spent: {spent}");
    println!("best: {best:.10}");
    println!("full_quenches: {}", full_quenches.load(Ordering::Relaxed));
    let hit = hit.into_inner().expect("hit");
    println!("hit: {}", hit.is_some());
    match hit {
        Some(h) => {
            println!("hit_energy: {:.10}", h.energy);
            println!("hit_calls: {}", h.calls);
            println!("hit_minimisations: {}", h.quenches);
            println!("hit_replica: {}", h.replica);
            println!("below_reference: {}", h.energy < reference - HIT_WINDOW);
        }
        None => {
            println!("hit_energy: none");
            println!("hit_calls: none");
            println!("hit_minimisations: none");
            println!("hit_replica: none");
            println!("below_reference: false");
        }
    }
}
