//! Time to target for LJ38, with and without the SOAP proposal.
//!
//! A global optimiser that reaches the minimum eventually is a Las
//! Vegas algorithm, and what characterises one is the distribution of
//! its time to a target rather than the energy it reports at whatever
//! budget was set. Mean final energy at a fixed budget rewards the arm
//! that is reliably mediocre and penalises the one that gambles, which
//! is exactly backwards for a funnel-exchange move.
//!
//! This runs independent searches, records the charged evaluations at
//! which each first reached the target, keeps the runs that never
//! arrived as censored rather than dropping them, and reports what the
//! sample supports: the success rates, whether their difference is
//! resolvable at this many runs, a shifted-exponential fit, and the
//! parallel speedup that fit implies for an ensemble.
//!
//! Usage: `ttt_lj38 [runs] [budget] [target]`, default 32 runs of
//! 25000 evaluations against -173.928427, the truncated octahedron.

use anneal_core::methods::cluster_hopping::{Config, Ledger, SoapProposalMode, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use anneal_core::runtime_distribution::{
    RunTime, parallel_speedup, reached_quantile, resolvable_shortfall, runs_for_resolution,
    shifted_exponential_fit, speedup_ceiling, success_rate,
};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const N: usize = 38;

fn ico_start(seed: u64) -> Array1<f64> {
    let sites =
        anneal_core::lattice::grow(&anneal_core::structure::Template::Icosahedral.points(), N);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(7));
    let mut x = Array1::zeros(3 * N);
    for (index, site) in sites.iter().take(N).enumerate() {
        for axis in 0..3 {
            let jitter = (rand::Rng::random::<f64>(&mut rng) - 0.5) * 0.1;
            x[3 * index + axis] = site[axis] + jitter;
        }
    }
    x
}

/// Charged evaluations at which this run first reached `target`, or
/// `None` when it never did.
fn time_to_target(soap: bool, seed: u64, budget: usize, target: f64) -> RunTime {
    let mut cfg = Config::recommended(N);
    cfg.soap_mode = if soap {
        SoapProposalMode::Flexible
    } else {
        SoapProposalMode::Off
    };
    let potential = PairPotential::lennard_jones(N);
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(v))
        });
        (value, relaxed)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let out = run_with_gradient(
        &cfg,
        ico_start(seed).view(),
        &mut ledger,
        &mut relax,
        None,
        &mut rng,
    );
    out.improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= target + 1e-4)
        .map(|(_, spent, _, _)| *spent as u64)
}

fn report(label: &str, runs: &[RunTime], processors: usize) -> usize {
    let hits = runs.iter().filter(|run| run.is_some()).count();
    let rate = success_rate(runs);
    println!("  {label}: reached {hits}/{} ({:.3})", runs.len(), rate);
    if let Some(median) = reached_quantile(runs, 0.5) {
        println!("    median time to target {median} evaluations");
    }
    match shifted_exponential_fit(runs) {
        Some((offset, lambda)) => {
            println!(
                "    shifted-exponential offset {offset:.0}, mean excess {:.0}",
                1.0 / lambda
            );
            println!(
                "    predicted speedup at {processors} replicas {:.2}, ceiling {:.2}",
                parallel_speedup(offset, lambda, processors),
                speedup_ceiling(offset, lambda)
            );
        }
        None => println!("    too few arrivals to fit a distribution"),
    }
    hits
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runs: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(32);
    let budget: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(25_000);
    let target: f64 = args
        .get(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(-173.928427);

    println!("LJ38 time to target {target} over {runs} runs of {budget} evaluations");
    let on: Vec<RunTime> = (0..runs as u64)
        .map(|seed| time_to_target(true, 300 + seed, budget, target))
        .collect();
    let off: Vec<RunTime> = (0..runs as u64)
        .map(|seed| time_to_target(false, 300 + seed, budget, target))
        .collect();

    let hits_on = report("soap on ", &on, 48);
    let hits_off = report("soap off", &off, 48);

    let slack = resolvable_shortfall(hits_on, hits_off, runs);
    println!(
        "\n  difference {} runs, resolvable at this sample {:.1}",
        hits_on as i64 - hits_off as i64,
        slack
    );
    let observed = success_rate(&on).max(success_rate(&off)).max(0.05);
    println!(
        "  runs needed to place a rate of {observed:.2}: {} to within 0.10, {} to within 0.05",
        runs_for_resolution(observed, 0.10),
        runs_for_resolution(observed, 0.05)
    );
    if (hits_on as f64 - hits_off as f64).abs() <= slack {
        println!("  the arms are not distinguishable at this many runs");
    }
}
