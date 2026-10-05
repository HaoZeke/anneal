//! Synchronous population basin hopping for LJ98.
//!
//! One iteration freezes the population, builds every child by a uniform
//! kick and one two-phase local search, then inserts children one at a
//! time. A child replaces its nearest member when it is inside the cutoff
//! and strictly better. Otherwise, when it is outside the cutoff, it
//! replaces the worst member when it is strictly better. The parent stays
//! unless it is the member that rule selects. A step that loses to its
//! parent can still enter by replacing someone else.
//!
//! Lengths in the printed schedule are in units of the pair minimum. The
//! potential below has its minimum at `2^(1/6)`, and the kick, the diameter,
//! and the two shell radii are multiplied by that factor once.
//!
//! ```text
//! lj98_paper_pbh self-test
//! lj98_paper_pbh <runs> <seed0> <anneal|fixed>
//! ```
//!
//! `anneal` multiplies the cutoff by 0.85 after iterations 50, 100, 150,
//! 200, and 250. `fixed` leaves the cutoff at 1.5 times the mean pairwise
//! shell distance of the initial population. A run stops at the first
//! two-phase search whose energy is within `1e-3` of `-543.665361`, or at
//! 1500 iterations. The printed call count is those moves. Each initial
//! member is a plain local minimum; the two-phase move is not used to
//! build the population.

use std::env;
use std::fs::OpenOptions;
use std::io::{Write, stdout};
use std::thread;
use std::time::Instant;

use anneal_core::diversity::DiversityAnnealer;
use anneal_core::methods::bank::{Admission, Bank};
use anneal_core::methods::cluster_hopping::random_cluster;
use anneal_core::methods::csa_cluster::coordination_histogram_distance;
use anneal_core::methods::two_phase::penalty;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N: usize = 98;
const K: usize = 40;
const MAX_STEP: u64 = 1500;
const TARGET: f64 = -543.665361;
const HIT: f64 = 1e-3;
const REFERENCE: f64 = -543.665361;

fn pair_unit() -> f64 {
    2f64.powf(1.0 / 6.0)
}

struct Point {
    energy: f64,
    x: Vec<f64>,
}

/// Shell distance at 1.25 and 1.55 pair-minimum units.
fn shell_distance(a: &[f64], b: &[f64]) -> f64 {
    let unit = pair_unit();
    coordination_histogram_distance(
        ArrayView1::from(a),
        ArrayView1::from(b),
        1.25 * unit,
        1.55 * unit,
    )
}

/// Cutoff in force during iteration `iteration` (1-based).
///
/// The factor 0.85 is applied after each of iterations 50, 100, 150, 200,
/// and 250, and the next iteration sees it.
fn dcut_for_iteration(iteration: u64, dcut0: f64, anneal: bool) -> f64 {
    if !anneal {
        return dcut0;
    }
    let applied = iteration.saturating_sub(1) / 50;
    dcut0 * 0.85_f64.powi(applied.min(5) as i32)
}

/// Index replaced, and whether that member was inside the cutoff.
fn replacement(energies: &[f64], dists: &[f64], child: f64, dcut: f64) -> Option<(usize, bool)> {
    let mut q = 0usize;
    for i in 1..dists.len() {
        if dists[i] < dists[q] {
            q = i;
        }
    }
    if dists[q] < dcut {
        if child < energies[q] {
            return Some((q, true));
        }
        return None;
    }
    let mut worst = 0usize;
    for i in 1..energies.len() {
        if energies[i] > energies[worst] {
            worst = i;
        }
    }
    if child < energies[worst] {
        return Some((worst, false));
    }
    None
}

fn emit(line: &str) {
    println!("{line}");
    let _ = stdout().flush();
    let Ok(dir) = env::var("PBH_LOGDIR") else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{dir}/progress.log"))
    {
        let _ = writeln!(file, "{line}");
    }
}

fn inf_norm(g: &Array1<f64>) -> f64 {
    g.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn two_phase(pot: &PairPotential, opt: &mut WarmLbfgs, x0: &[f64]) -> (f64, Vec<f64>, usize, f64) {
    let cutoff = 3.5 * pair_unit();
    let start = Array1::from_vec(x0.to_vec());
    opt.forget();
    let (_biased, biased, n1) = opt.minimize(start.view(), 2000, |v| {
        let (e, mut g) = pot.value_and_gradient(v);
        let (pe, pg) = penalty(v, cutoff, 1.0, 0.0);
        for i in 0..g.len() {
            g[i] += pg[i];
        }
        Some((e + pe, g))
    });
    opt.forget();
    let (energy, plain, n2) =
        opt.minimize(biased.view(), 2000, |v| Some(pot.value_and_gradient(v)));
    let (_, g) = pot.value_and_gradient(plain.view());
    let ginf = inf_norm(&g);
    (energy, plain.to_vec(), n1 + n2, ginf)
}

struct Run {
    seed: u64,
    hit: bool,
    calls: u64,
    init_calls: u64,
    energy: f64,
    iteration: u64,
    near: u64,
    far: u64,
    unconverged: u64,
}

fn shell_view(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let unit = pair_unit();
    coordination_histogram_distance(a, b, 1.25 * unit, 1.55 * unit)
}

/// Plain local minimum. The two-phase move is the iteration, not the seed.
fn plain_local(pot: &PairPotential, opt: &mut WarmLbfgs, rng: &mut StdRng) -> (Point, usize, f64) {
    let seed = random_cluster(N, 0.7, 0.85, rng);
    opt.forget();
    let (energy, x, evals) = opt.minimize(seed.view(), 2000, |v| Some(pot.value_and_gradient(v)));
    let (_, g) = pot.value_and_gradient(x.view());
    (
        Point {
            energy,
            x: x.to_vec(),
        },
        evals,
        inf_norm(&g),
    )
}

fn run_one(seed: u64, anneal: bool) -> Run {
    let mut rng = StdRng::seed_from_u64(seed);
    let pot = PairPotential::lennard_jones(N);
    let mut opt = WarmLbfgs::default();
    let mut bank = Bank::new(K, 1.0);
    let mut init_calls = 0u64;
    let mut unconverged = 0u64;
    let mut best = f64::INFINITY;
    let started = Instant::now();
    for _ in 0..K {
        let (point, _, ginf) = plain_local(&pot, &mut opt, &mut rng);
        if ginf > 1e-4 {
            unconverged += 1;
        }
        init_calls += 1;
        best = best.min(point.energy);
        if point.energy <= TARGET + HIT {
            emit(&format!(
                "seed {seed} hit 1 calls 0 init {init_calls} energy {:.6}",
                point.energy
            ));
            return Run {
                seed,
                hit: true,
                calls: 0,
                init_calls,
                energy: point.energy,
                iteration: 0,
                near: 0,
                far: 0,
                unconverged,
            };
        }
        let stored = Array1::from_vec(point.x);
        assert!(
            bank.seed(stored.view(), point.energy),
            "the initial population has one slot per member"
        );
    }
    let coords: Vec<Vec<f64>> = bank
        .members()
        .iter()
        .map(|member| member.state.to_vec())
        .collect();
    let slots: Vec<usize> = (0..coords.len()).collect();
    let dcut0 = DiversityAnnealer::scaled_from_population(
        &slots,
        |i, j| shell_distance(&coords[i], &coords[j]),
        1.5,
    )
    .map(|schedule| schedule.initial())
    .unwrap_or(1.0);
    bank.dcut = dcut0;
    emit(&format!(
        "seed {seed} init {init_calls} best {best:.4} seconds {:.1} dcut {dcut0:.3}",
        started.elapsed().as_secs_f64()
    ));
    let half = 0.4 * pair_unit();
    let mut calls = 0u64;
    let mut near = 0u64;
    let mut far = 0u64;
    for iteration in 1..=MAX_STEP {
        bank.dcut = dcut_for_iteration(iteration, dcut0, anneal);
        let parents: Vec<Vec<f64>> = bank
            .members()
            .iter()
            .map(|member| member.state.to_vec())
            .collect();
        let mut children = Vec::with_capacity(K);
        let mut hit_at: Option<Point> = None;
        for parent in &parents {
            let kicked = parent
                .iter()
                .map(|v| v + rng.random_range(-half..half))
                .collect::<Vec<_>>();
            let (energy, x, _, ginf) = two_phase(&pot, &mut opt, &kicked);
            calls += 1;
            if ginf > 1e-4 {
                unconverged += 1;
            }
            best = best.min(energy);
            let child = Point { energy, x };
            if energy <= TARGET + HIT {
                hit_at = Some(child);
                break;
            }
            children.push(child);
        }
        if let Some(child) = hit_at {
            emit(&format!(
                "seed {seed} hit 1 calls {calls} init {init_calls} energy {:.6} iteration {iteration}",
                child.energy
            ));
            return Run {
                seed,
                hit: true,
                calls,
                init_calls,
                energy: child.energy,
                iteration,
                near,
                far,
                unconverged,
            };
        }
        for child in children {
            let stored = Array1::from_vec(child.x);
            match bank.offer(stored.view(), child.energy, shell_view) {
                Admission::Improved(_) => near += 1,
                Admission::Displaced(_) => far += 1,
                Admission::Duplicate(_) | Admission::Added(_) | Admission::Rejected => {}
            }
        }
        emit(&format!(
            "seed {seed} iter {iteration} calls {calls} best {best:.4} near {near} far {far}"
        ));
    }
    Run {
        seed,
        hit: false,
        calls,
        init_calls,
        energy: best,
        iteration: MAX_STEP,
        near,
        far,
        unconverged,
    }
}

fn self_test() {
    let unit = pair_unit();
    let s = unit / (8.0_f64).sqrt();
    let tetra = |shift: f64| -> Vec<f64> {
        let raw = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let mut x = Vec::with_capacity(12);
        for p in raw {
            for c in p {
                x.push(c * s + shift);
            }
        }
        x
    };
    let a = tetra(0.0);
    let b = tetra(3.0);
    assert!(
        shell_distance(&a, &b) == 0.0,
        "a rigid shift is the same cluster"
    );
    let mut swapped = a.clone();
    swapped.rotate_left(3);
    assert!(
        shell_distance(&a, &swapped) == 0.0,
        "an atom permutation is the same cluster"
    );
    let mut broken = a.clone();
    for k in 0..3 {
        broken[9 + k] += 10.0 * unit;
    }
    let d = shell_distance(&a, &broken);
    assert!(
        (d - 36.0).abs() < 1e-9,
        "one atom pulled out of both shells changes the histogram by 36, got {d}"
    );

    // Parent is index 0. The child is nearer to index 1 and better than both,
    // so index 1 changes and the parent stays.
    let energies = [-10.0, -9.0];
    let dists = [0.2, 0.05];
    assert_eq!(replacement(&energies, &dists, -11.0, 1.0), Some((1, true)));
    // Uphill from the parent, still better than the nearer neighbour.
    assert_eq!(replacement(&energies, &dists, -9.5, 1.0), Some((1, true)));
    // Near the parent and worse than the parent: discarded.
    assert_eq!(replacement(&[-10.0, -9.0], &[0.05, 0.2], -9.5, 1.0), None);
    // Outside every cutoff and better than the worst member.
    assert_eq!(
        replacement(&[-10.0, -7.0], &[5.0, 4.0], -8.0, 1.0),
        Some((1, false))
    );
    // Outside every cutoff and worse than the worst member.
    assert_eq!(replacement(&[-10.0, -7.0], &[5.0, 4.0], -6.0, 1.0), None);

    let d0 = 2.0;
    assert!((dcut_for_iteration(1, d0, true) - 2.0).abs() < 1e-12);
    assert!((dcut_for_iteration(50, d0, true) - 2.0).abs() < 1e-12);
    assert!((dcut_for_iteration(51, d0, true) - 2.0 * 0.85).abs() < 1e-12);
    let fifth = 2.0 * 0.85_f64.powi(5);
    assert!((dcut_for_iteration(251, d0, true) - fifth).abs() < 1e-12);
    assert!((dcut_for_iteration(1000, d0, true) - fifth).abs() < 1e-12);
    assert!((dcut_for_iteration(1000, d0, false) - 2.0).abs() < 1e-12);

    let _ = ArrayView1::from(&[0.0_f64, 0.0, 0.0][..]);
    println!("self-test ok");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("self-test") {
        self_test();
        return;
    }
    let runs: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(10);
    let seed0: u64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mode = args.get(2).map(String::as_str).unwrap_or("anneal");
    let anneal = mode != "fixed";
    let unit = pair_unit();
    emit(&format!(
        "LJ98 synchronous population K={K} MaxStep={MAX_STEP} mode={mode} seeds {seed0}..{} D={:.6} kick={:.6} shells {:.6} {:.6} target {REFERENCE} tol {HIT}",
        seed0 + runs as u64 - 1,
        3.5 * unit,
        0.4 * unit,
        1.25 * unit,
        1.55 * unit,
    ));
    let quench_started = Instant::now();
    let pot = PairPotential::lennard_jones(N);
    let mut opt = WarmLbfgs::default();
    let mut rng = StdRng::seed_from_u64(0);
    let sample = random_cluster(N, 0.7, 0.85, &mut rng);
    let (energy, _, evals, ginf) = two_phase(
        &pot,
        &mut opt,
        sample.as_slice().expect("sample is contiguous"),
    );
    emit(&format!(
        "one_quench {:.3}s evals {evals} ginf {ginf:.3e} energy {energy:.4}",
        quench_started.elapsed().as_secs_f64()
    ));
    let mut handles = Vec::with_capacity(runs);
    for i in 0..runs {
        let seed = seed0 + i as u64;
        handles.push(thread::spawn(move || run_one(seed, anneal)));
    }
    let mut done = Vec::with_capacity(runs);
    for handle in handles {
        done.push(handle.join().expect("run"));
    }
    done.sort_by_key(|run| run.seed);
    let mut hits = 0usize;
    let mut hit_calls = 0u64;
    for run in &done {
        if run.hit {
            hits += 1;
            hit_calls += run.calls;
        }
        emit(&format!(
            "seed {} hit {} calls {} init {} energy {:.6} iteration {} near {} far {} unconverged {}",
            run.seed,
            run.hit as u8,
            run.calls,
            run.init_calls,
            run.energy,
            run.iteration,
            run.near,
            run.far,
            run.unconverged,
        ));
    }
    let mean = if hits == 0 {
        f64::NAN
    } else {
        hit_calls as f64 / hits as f64
    };
    emit(&format!("RS {hits}/{runs} mean_hit_calls {mean:.1}"));
}
