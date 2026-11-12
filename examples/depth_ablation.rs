// Copyright (c) anneal developers
//! One seed of the paper preset, with or without the depth posterior.
//!
//! ```text
//! depth_ablation <lj38|lj55|lj75|lj98|lj104> <seed> <posterior|uniform> [budget]
//! ```
//!
//! `posterior` is [`Config::recommended`] unchanged. `uniform` is that same
//! preset with [`ArmSelection::Uniform`]: arms are drawn uniformly, and every
//! other control stays. The budget argument is optional and is not used by
//! the paper campaign. Omitting it charges [`paper_budget`].
//!
//! A hit is a quenched energy within 0.001 of the Cambridge minimum. The
//! search stops there. Stdout is one CSV row and nothing else.

use anneal_core::methods::cluster_hopping::{
    ArmSelection, Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match run(&args) {
        Ok(status) => status,
        Err(err) => {
            eprintln!("{err}");
            2
        }
    };
    std::process::exit(status);
}

fn run(args: &[String]) -> Result<i32, String> {
    let system = args.first().ok_or("need a system")?.as_str();
    let seed: u64 = args
        .get(1)
        .and_then(|text| text.parse().ok())
        .ok_or("need a seed")?;
    let arm = args.get(2).ok_or("need an arm")?.as_str();
    let spec = system_spec(system).ok_or_else(|| format!("unknown system {system}"))?;
    let budget = match args.get(3) {
        Some(text) => text
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("budget {text} is not a positive integer"))?,
        None => spec.budget,
    };
    let cfg = configured(spec.n, arm)?;
    let start = start_of(&cfg, spec.n, seed);
    let (outcome, wall) = search(&cfg, spec.n, &start, seed, budget, spec.reference);
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let calls = if hit {
        at_hit.unwrap_or(outcome.charged).to_string()
    } else {
        String::new()
    };
    println!(
        "{system},{arm},{seed},{budget},{},{calls},{},{:.8},{wall:.6}",
        usize::from(hit),
        outcome.charged,
        outcome.best,
    );
    Ok(0)
}

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    let (n, reference) = match name {
        "lj38" => (38, -173.928427),
        "lj55" => (55, -279.248470),
        "lj75" => (75, -397.492331),
        "lj98" => (98, -543.665361),
        "lj104" => (104, -582.086642),
        _ => return None,
    };
    Some(Spec {
        n,
        budget: paper_budget(n)?,
        reference,
    })
}

/// Budgets from `scripts/elja_submit_jcc_lj.sh` (`PAPER_BUDGETS`).
fn paper_budget(n: usize) -> Option<usize> {
    match n {
        38 => Some(400_000),
        55 => Some(1_000_000),
        75 | 98 | 104 => Some(4_000_000),
        _ => None,
    }
}

fn configured(n: usize, arm: &str) -> Result<Config, String> {
    let mut cfg = Config::recommended(n);
    if cfg.arm_selection != ArmSelection::Posterior || !cfg.depth_reward || !cfg.allocate_moves {
        return Err("recommended preset is not the depth posterior".into());
    }
    match arm {
        "posterior" => Ok(cfg),
        "uniform" => {
            cfg.arm_selection = ArmSelection::Uniform;
            Ok(cfg)
        }
        _ => Err(format!("unknown arm {arm}")),
    }
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn search(
    cfg: &Config,
    n: usize,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
) -> (anneal_core::methods::cluster_hopping::Outcome, f64) {
    let potential = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
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
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(v).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    (outcome, started.elapsed().as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::{configured, paper_budget, system_spec};
    use anneal_core::methods::cluster_hopping::ArmSelection;

    #[test]
    fn paper_budgets_are_the_elja_map() {
        assert_eq!(paper_budget(38), Some(400_000));
        assert_eq!(paper_budget(55), Some(1_000_000));
        assert_eq!(paper_budget(75), Some(4_000_000));
        assert_eq!(paper_budget(98), Some(4_000_000));
        assert_eq!(paper_budget(104), Some(4_000_000));
        assert_eq!(system_spec("lj38").unwrap().budget, 400_000);
        assert_eq!(system_spec("lj55").unwrap().budget, 1_000_000);
        assert_eq!(system_spec("lj75").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("lj98").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("lj104").unwrap().budget, 4_000_000);
        assert!((system_spec("lj38").unwrap().reference + 173.928427).abs() < 1e-12);
        assert!((system_spec("lj55").unwrap().reference + 279.248470).abs() < 1e-12);
        assert!((system_spec("lj75").unwrap().reference + 397.492331).abs() < 1e-12);
        assert!((system_spec("lj98").unwrap().reference + 543.665361).abs() < 1e-12);
        assert!((system_spec("lj104").unwrap().reference + 582.086642).abs() < 1e-12);
    }

    #[test]
    fn uniform_changes_only_the_arm_rule() {
        let posterior = configured(75, "posterior").unwrap();
        let uniform = configured(75, "uniform").unwrap();
        assert_eq!(posterior.arm_selection, ArmSelection::Posterior);
        assert_eq!(uniform.arm_selection, ArmSelection::Uniform);
        assert!(posterior.depth_reward && uniform.depth_reward);
        assert!(posterior.allocate_moves && uniform.allocate_moves);
        assert_eq!(posterior.move_library, uniform.move_library);
        assert_eq!(posterior.temperature, uniform.temperature);
        assert_eq!(posterior.screen_steps, uniform.screen_steps);
        assert_eq!(posterior.relax_steps, uniform.relax_steps);
        assert_eq!(posterior.return_screen, uniform.return_screen);
        assert_eq!(posterior.surfaces.len(), uniform.surfaces.len());
        assert!(configured(38, "thompson").is_err());
    }
}
