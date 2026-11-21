//! The working-set check asks the allocator for at least what the first hop
//! of a search holds.
//!
//! `memory::first_hop_bytes` is a sum over the vectors and structures that a
//! search keeps alive through its opening quench and first hop. This test
//! runs that hop for a small `n` under an allocator that counts live bytes,
//! and compares the peak with the estimate for the same `n`. A new vector
//! that the hop holds, or one that grows with `n` faster than the estimate
//! says, makes the estimate smaller than the peak and fails here.
//!
//! One test function drives every case, because the counter is process wide
//! and the test harness runs functions side by side.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::memory::first_hop_bytes;
use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::{search, search_from};
use anneal_core::potentials::random_tip4p_cluster as random_cluster;
use anneal_core::potentials::{PairPotential, Tip4pCluster};
use eindir_core::Bounds;
use eindir_core::Objective;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let at = unsafe { System.alloc(layout) };
        if !at.is_null() {
            grew(layout.size());
        }
        at
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let at = unsafe { System.alloc_zeroed(layout) };
        if !at.is_null() {
            grew(layout.size());
        }
        at
    }

    unsafe fn dealloc(&self, at: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(at, layout) }
    }

    unsafe fn realloc(&self, at: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(at, layout, size) };
        if !moved.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            grew(size);
        }
        moved
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// An objective that costs what a Python callable costs: the argument is a
/// copy of the state, the answer comes back as a sequence of numbers, and the
/// gradient is built from that sequence.
struct AsCallable<P>(P);

impl<P: DifferentiableObjective<f64>> Objective<f64> for AsCallable<P> {
    fn dim(&self) -> usize {
        Objective::dim(&self.0)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.0.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl<P: DifferentiableObjective<f64>> Gradient<f64> for AsCallable<P> {
    fn dim(&self) -> usize {
        Objective::dim(&self.0)
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl<P: DifferentiableObjective<f64>> DifferentiableObjective<f64> for AsCallable<P> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let argument = black_box(x.to_vec());
        let (energy, gradient) = self.0.value_and_gradient(ArrayView1::from(&argument));
        let sequence = black_box(gradient.to_vec());
        let built = Array1::from_iter(sequence.iter().copied());
        black_box((&argument, &sequence));
        (energy, built)
    }
}

/// Bytes one call of the objective holds beyond the three vectors that
/// `AsCallable` makes: the objective's own working memory, which belongs to
/// whoever wrote it and not to the search.
fn call_bytes<P: DifferentiableObjective<f64>>(
    objective: &AsCallable<P>,
    x: &Array1<f64>,
) -> usize {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    black_box(objective.value_and_gradient(x.view()));
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    peak.saturating_sub(3 * 8 * x.len())
}

enum Case {
    Atoms(&'static str, fn(usize) -> Config),
    Waters(&'static str, fn(usize) -> Config),
}

impl Case {
    fn label(&self) -> &'static str {
        match self {
            Self::Atoms(label, _) | Self::Waters(label, _) => label,
        }
    }

    fn per_point(&self) -> usize {
        match self {
            Self::Atoms(..) => 3,
            Self::Waters(..) => 6,
        }
    }

    /// What the objective itself holds during a call.
    fn objective_bytes(&self, n: usize) -> usize {
        match self {
            Self::Atoms(..) => {
                let objective = AsCallable(PairPotential::lennard_jones(n));
                let x = Array1::from_elem(3 * n, 0.5);
                call_bytes(&objective, &x)
            }
            Self::Waters(..) => {
                let objective = AsCallable(Tip4pCluster::new(n));
                let mut rng = StdRng::seed_from_u64(3);
                call_bytes(&objective, &random_cluster(n, &mut rng))
            }
        }
    }

    /// Peak live bytes of a run of `budget` calls, and the hops it made.
    fn run(&self, n: usize, budget: usize) -> (usize, usize) {
        match self {
            Self::Atoms(_, config) => {
                let objective = AsCallable(PairPotential::lennard_jones(n));
                measured(&objective, &config(n), None, budget)
            }
            Self::Waters(_, config) => {
                let objective = AsCallable(Tip4pCluster::new(n));
                let mut rng = StdRng::seed_from_u64(3);
                let start = random_cluster(n, &mut rng);
                measured(&objective, &config(n), Some(&start), budget)
            }
        }
    }
}

fn measured<O>(
    objective: &O,
    config: &Config,
    start: Option<&Array1<f64>>,
    budget: usize,
) -> (usize, usize)
where
    O: DifferentiableObjective<f64>,
{
    let mut ledger = Ledger::new(budget);
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = match start {
        None => search(objective, config, &mut ledger, 1),
        Some(start) => search_from(objective, config, &mut ledger, start.view(), 1),
    };
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let (outcome, _) = result.expect("a finite start");
    (peak, outcome.hops)
}

/// The most live bytes over every budget whose run ends with exactly one hop.
///
/// A run that ends in its first hop has made the opening quench and that hop
/// in full, and a longer budget holds at least what a shorter one did, so
/// the largest peak among these runs is the first hop's.
fn first_hop_peak(case: &Case, n: usize) -> usize {
    let mut peak = None;
    let mut budget = 20;
    loop {
        let (bytes, hops) = case.run(n, budget);
        match hops {
            0 => {}
            1 => peak = Some(peak.map_or(bytes, |seen: usize| seen.max(bytes))),
            _ => break,
        }
        budget += 10;
        assert!(
            budget < 40_000,
            "{} n = {n} never made a second hop",
            case.label()
        );
    }
    peak.unwrap_or_else(|| panic!("{} n = {n} never ended in its first hop", case.label()))
}

#[test]
fn the_estimate_is_at_least_what_the_first_hop_holds() {
    anneal_core::install_rgpot_kernels!();
    let cases = [
        Case::Atoms("recommended", Config::recommended),
        Case::Atoms("for_cluster", Config::for_cluster),
        Case::Atoms("derived", Config::derived),
        Case::Atoms("communicating", Config::communicating),
        Case::Waters("for_tip4p", Config::for_tip4p),
        Case::Waters("recommended_tip4p", Config::recommended_tip4p),
    ];
    let mut report = Vec::new();
    let mut short = Vec::new();
    let mut loose = Vec::new();
    for case in &cases {
        for n in [3usize, 4, 8, 13, 21, 38, 55] {
            if matches!(case, Case::Waters(..)) && n > 13 {
                continue;
            }
            let held = first_hop_peak(case, n);
            let len = case.per_point() * n;
            let estimate = first_hop_bytes(n, len).expect("a small n does not overflow")
                + case.objective_bytes(n);
            report.push(format!(
                "{:>18} n = {n:>2}: held {held:>7} B, estimated {estimate:>7} B, ratio {:.2}",
                case.label(),
                estimate as f64 / held as f64
            ));
            if estimate < held {
                short.push(report.last().cloned().unwrap_or_default());
            }
            if n >= 21 && estimate > 6 * held {
                loose.push(report.last().cloned().unwrap_or_default());
            }
        }
    }
    println!("{}", report.join("\n"));
    assert!(
        short.is_empty(),
        "the estimate is below what the first hop holds:\n{}",
        short.join("\n")
    );
    assert!(
        loose.is_empty(),
        "the estimate is more than six times what the first hop holds, which \
         would refuse searches the machine can run:\n{}",
        loose.join("\n")
    );
}
