//! Paper control drivers for the registered seed blocks.
//!
//! `recommended` is `Config::recommended`. `for_cluster` is the screened
//! basin-hopping control: temperature `0.8`, screen of 25 steps, the Bayes
//! screen on, and threshold `7/15`. It is also the in-tree control where the outline
//! leaves GMIN on the optbench side. `derived` is `Config::derived`.
//! `soap-on` and `soap-off` share `Config::recommended` and differ only
//! in `SoapProposalMode`. `soap38` starts from the icosahedron used by
//! `ttt_lj38`. `lj12-groups` is named and refused: four water-shaped
//! groups are not deposited here.
//!
//! `PAPER_SMOKE=1` runs the first registered seed at 200 force evaluations
//! and writes a file whose name contains `SMOKE`. That file is not a paper
//! table. Full runs launch from the v0.12.0 tag.
//!
//! ```text
//! paper_controls <system> <arm>
//! ```

use anneal_core::catalog::lj::system_signature;
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, SoapProposalMode, optimize_with_gradient, run_with_gradient,
};
use anneal_core::methods::population_annealing::{
    PopulationAnneal, PopulationReport, population_anneal,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::run_manifest::RunManifest;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const HIT: f64 = 1e-4;
const SMOKE_CALLS: usize = 200;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Recommended,
    ForCluster,
    Derived,
    SoapOn,
    SoapOff,
    /// Hukushima–Iba population annealing. Not a `Config` hop.
    Population,
}

struct Campaign {
    n: usize,
    rho: Option<f64>,
    seed0: u64,
    seed_count: u64,
    paper_budget: Option<usize>,
    reference: Option<f64>,
    icosahedral: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let system = args.first().map(String::as_str).unwrap_or("lj38");
    let arm_name = args.get(1).map(String::as_str).unwrap_or("recommended");
    let smoke = std::env::var("PAPER_SMOKE").ok().as_deref() == Some("1");
    let campaign = match campaign(system) {
        Ok(campaign) => campaign,
        Err(reason) => {
            eprintln!("{reason}");
            std::process::exit(2);
        }
    };
    let arm = match parse_arm(arm_name) {
        Some(arm) => arm,
        None => {
            eprintln!("unknown arm {arm_name}");
            std::process::exit(2);
        }
    };
    let budget = if smoke {
        SMOKE_CALLS
    } else {
        match campaign.paper_budget {
            Some(budget) => budget,
            None => {
                eprintln!("paper budget is unset for {system}");
                std::process::exit(2);
            }
        }
    };
    let seeds = if smoke { 1 } else { campaign.seed_count };
    let cfg = apply_arm(campaign.n, arm);
    let config_digest = cfg.resolved_sha256().expect("config digest");
    let (engine_kind, engine_digest) = engine_identity(campaign.n, campaign.rho);
    let mut report = String::new();
    push(
        &mut report,
        &format!(
            "label: {}\ndriver: paper_controls\ncommit: {}\ncpu_model: {}\n\
             system: {system}\narm: {arm_name}\nn: {}\nseed_start: {}\nseed_count: {seeds}\n\
             paper_seed_count: {}\npaper_budget: {}\nsmoke_budget: {SMOKE_CALLS}\n\
             config_digest: {config_digest}\nengine_kind: {engine_kind}\n\
             engine_digest: {engine_digest}\nreference: {}\nhit_tolerance: {HIT}\n",
            if smoke { "smoke" } else { "untagged" },
            git_commit(),
            cpu_model(),
            campaign.n,
            campaign.seed0,
            campaign.seed_count,
            campaign
                .paper_budget
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unset".into()),
            campaign
                .reference
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "unset".into()),
        ),
    );
    for offset in 0..seeds {
        let seed = campaign.seed0 + offset;
        let (charged, best, wall, force_to_hit, generations) = if arm == Arm::Population {
            let (report, charged, wall, force_to_hit, generations) =
                run_population(&campaign, seed, budget, smoke);
            (charged, report.best_energy, wall, force_to_hit, generations)
        } else {
            let (outcome, wall, force_to_hit) = run_seed(&cfg, &campaign, seed, budget);
            let force_to_hit = force_to_hit.filter(|_| {
                campaign
                    .reference
                    .is_some_and(|reference| outcome.best <= reference + HIT)
            });
            (outcome.charged, outcome.best, wall, force_to_hit, 0)
        };
        let hit = match campaign.reference {
            None => "unscored",
            Some(reference) if best <= reference + HIT => "true",
            Some(_) => "false",
        };
        push(
            &mut report,
            &format!(
                "seed: {seed}\nforce_count: {charged}\nforce_to_hit: {}\nwall_seconds: {wall:.6}\n\
                 best: {best:.6}\nhit: {hit}\ngenerations: {generations}\n",
                force_to_hit
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".into()),
            ),
        );
    }
    let path = output_path(smoke);
    let mut manifest = RunManifest::new(
        format!("paper-controls-{system}-{arm_name}"),
        campaign.seed0,
        budget as u64,
    );
    manifest.engine.build_identity = Some(git_commit());
    manifest.add_artifact("log", report.as_bytes());
    let manifest_json = manifest.to_json().expect("manifest");
    let manifest_path = format!("{path}.manifest.json");
    if smoke && !manifest_path.contains("SMOKE") {
        panic!("PAPER_SMOKE manifest path must contain SMOKE");
    }
    std::fs::write(&path, &report).expect("control log");
    std::fs::write(&manifest_path, manifest_json).expect("manifest");
    print!("{report}");
}

fn campaign(name: &str) -> Result<Campaign, String> {
    if name == "lj12-groups" {
        return Err("four water-shaped LJ groups are not deposited in this binary".into());
    }
    if let Some(rest) = name.strip_prefix("morse") {
        let (n_text, rho_text) = rest
            .split_once('-')
            .ok_or_else(|| format!("unknown system {name}"))?;
        let n: usize = n_text
            .parse()
            .map_err(|_| format!("unknown system {name}"))?;
        let rho: f64 = rho_text
            .parse()
            .map_err(|_| format!("unknown system {name}"))?;
        if ![38, 55, 80].contains(&n) || ![6.0, 10.0, 14.0].contains(&rho) {
            return Err(format!("unknown system {name}"));
        }
        return Ok(Campaign {
            n,
            rho: Some(rho),
            seed0: 2200,
            seed_count: 48,
            paper_budget: None,
            reference: morse_reference(n, rho),
            icosahedral: false,
        });
    }
    let (n, seed0, seed_count, paper_budget, icosahedral) = match name {
        "lj38" => (38, 2000, 72, Some(400_000), false),
        "lj75" => (75, 2000, 48, Some(4_000_000), false),
        "lj98" => (98, 2000, 48, Some(4_000_000), false),
        "lj31" => (31, 2200, 48, None, false),
        "lj76" => (76, 2200, 48, None, false),
        "lj102" => (102, 2200, 48, None, false),
        "soap38" => (38, 5000, 64, Some(25_000), true),
        _ => return Err(format!("unknown system {name}")),
    };
    Ok(Campaign {
        n,
        rho: None,
        seed0,
        seed_count,
        paper_budget,
        reference: lj_reference(n),
        icosahedral,
    })
}

fn parse_arm(name: &str) -> Option<Arm> {
    Some(match name {
        "recommended" => Arm::Recommended,
        "for_cluster" => Arm::ForCluster,
        "derived" => Arm::Derived,
        "soap-on" => Arm::SoapOn,
        "soap-off" => Arm::SoapOff,
        "population" => Arm::Population,
        _ => return None,
    })
}

fn apply_arm(n: usize, arm: Arm) -> Config {
    match arm {
        Arm::Recommended => Config::recommended(n),
        Arm::ForCluster => Config::for_cluster(n),
        Arm::Derived => Config::derived(n),
        Arm::SoapOn => {
            let mut cfg = Config::recommended(n);
            // The 38-point recommended library is the named-lattice reseed.
            // The soap ablation is the LeanBurst stack with the proposal on.
            cfg.move_library = MoveLibrary::LeanBurst;
            cfg.soap_mode = SoapProposalMode::Flexible;
            cfg
        }
        Arm::SoapOff => {
            let mut cfg = Config::recommended(n);
            cfg.move_library = MoveLibrary::LeanBurst;
            cfg.soap_mode = SoapProposalMode::Off;
            cfg
        }
        // The population arm does not hop. The digest below is unused.
        Arm::Population => Config::for_cluster(n),
    }
}

/// Hukushima–Iba population with Machta's half-population ESS step.
///
/// A smoke uses four walkers so the 200-call budget still reaches a
/// quench. A full run uses the reference population of 100.
fn run_population(
    campaign: &Campaign,
    seed: u64,
    budget: usize,
    smoke: bool,
) -> (PopulationReport, usize, f64, Option<usize>, usize) {
    let mut cfg = PopulationAnneal::reference(campaign.n);
    if smoke {
        cfg.walkers = 4;
        cfg.quench_reserve_per_walker = 30;
        cfg.quench_steps = 40;
    }
    let kind = match campaign.rho {
        Some(rho) => PairKind::Morse { rho },
        None => PairKind::LennardJones,
    };
    let potential = match kind {
        PairKind::LennardJones => PairPotential::lennard_jones(campaign.n),
        PairKind::Morse { rho } => PairPotential::morse(campaign.n, rho),
    };
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let reference = campaign.reference;
    let mut first_hit = None;
    let started = Instant::now();
    let report = population_anneal(
        campaign.n,
        &mut ledger,
        &cfg,
        |led, x| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(x).0)
        },
        |led, x, steps| {
            optimiser.forget();
            let (value, relaxed, _) = optimiser.minimize(x, steps, |v| {
                if !led.charge() {
                    return None;
                }
                Some(potential.value_and_gradient(v))
            });
            if first_hit.is_none() && reference.is_some_and(|target| value <= target + HIT) {
                first_hit = Some(led.spent());
            }
            value.is_finite().then_some((value, relaxed))
        },
        &mut rng,
    );
    (
        report.clone(),
        ledger.spent(),
        started.elapsed().as_secs_f64(),
        first_hit,
        report.generations,
    )
}

fn run_seed(
    cfg: &Config,
    campaign: &Campaign,
    seed: u64,
    budget: usize,
) -> (
    anneal_core::methods::cluster_hopping::Outcome,
    f64,
    Option<usize>,
) {
    let kind = match campaign.rho {
        Some(rho) => PairKind::Morse { rho },
        None => PairKind::LennardJones,
    };
    let potential = match kind {
        PairKind::LennardJones => PairPotential::lennard_jones(campaign.n),
        PairKind::Morse { rho } => PairPotential::morse(campaign.n, rho),
    };
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let reference = campaign.reference;
    // The outcome keeps 512 improvement records. A later crossing still
    // updates the best energy, so the hit clock is this quench.
    let first_hit = AtomicUsize::new(usize::MAX);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        let before = led.best;
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(v))
        });
        if first_hit.load(Ordering::Relaxed) == usize::MAX
            && reference.is_some_and(|target| value.is_finite() && value <= target + HIT)
            && value < before - 1e-10
        {
            first_hit.store(led.spent(), Ordering::Relaxed);
        }
        (value, relaxed)
    };
    // The climb reads this gradient. Without it the stall leave never runs
    // and the only call-clock response is a fresh start, which returns to
    // the Mackay icosahedron.
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(v).1)
    };
    let started = Instant::now();
    // One `StdRng`, the library entry. A second stream reseeded at `seed`
    // after the start draw is a different search.
    let outcome = if campaign.icosahedral {
        let start = ico_start(campaign.n, seed);
        let mut rng = StdRng::seed_from_u64(seed);
        run_with_gradient(
            cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut rng,
        )
    } else {
        optimize_with_gradient(cfg, &mut ledger, &mut relax, Some(&mut grad), seed)
    }
    .expect("finite start");
    let from_list = campaign.reference.and_then(|reference| {
        outcome
            .improvements
            .iter()
            .find(|(_, _, _, energy)| *energy <= reference + HIT)
            .map(|(_, spent, _, _)| *spent)
    });
    (
        outcome,
        started.elapsed().as_secs_f64(),
        match first_hit.load(Ordering::Relaxed) {
            usize::MAX => from_list,
            spent => Some(spent),
        },
    )
}

/// Icosahedral 38-mer perturbation from `examples/ttt_lj38.rs`.
fn ico_start(n: usize, seed: u64) -> Array1<f64> {
    let sites =
        anneal_core::lattice::grow(&anneal_core::structure::Template::Icosahedral.points(), n);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(7));
    let mut x = Array1::zeros(3 * n);
    for (index, site) in sites.iter().take(n).enumerate() {
        for axis in 0..3 {
            let jitter = (rng.random::<f64>() - 0.5) * 0.1;
            x[3 * index + axis] = site[axis] + jitter;
        }
    }
    x
}

/// Cambridge Cluster Database values already scored in this repository.
/// LJ31 and LJ76 are not among them.
fn lj_reference(n: usize) -> Option<f64> {
    match n {
        38 => Some(-173.928427),
        75 => Some(-397.492331),
        98 => Some(-543.665361),
        102 => Some(-569.363652),
        _ => None,
    }
}

/// Bold Doye-Wales entries copied from `examples/cluster_bench.rs`.
/// The rho = 14, N = 80 value is the 80G entry recorded by
/// `examples/lattice_ensemble_ablation.rs`.
fn morse_reference(n: usize, rho: f64) -> Option<f64> {
    match (n, rho as i64) {
        (38, 6) => Some(-157.477108),
        (55, 6) => Some(-250.286609),
        (38, 10) => Some(-145.849817),
        (55, 10) => Some(-225.814286),
        (38, 14) => Some(-144.321054),
        (55, 14) => Some(-220.646208),
        (80, 14) => Some(-340.811371),
        _ => None,
    }
}

fn engine_identity(n: usize, rho: Option<f64>) -> (String, String) {
    match rho {
        None => {
            let signature = system_signature(n).expect("LJ signature");
            (signature.engine.kind, hex(&signature.engine.config_digest))
        }
        Some(rho) => (
            "morse-doye-wales-v1".into(),
            sha256_hex(format!("morse epsilon=1 r0=1 rho={rho}").as_bytes()),
        ),
    }
}

fn output_path(smoke: bool) -> String {
    let path = std::env::var("PAPER_OUT").unwrap_or_else(|_| {
        if smoke {
            "paper-controls-SMOKE.txt".into()
        } else {
            "paper-controls.txt".into()
        }
    });
    if !require_smoke_name(smoke, &path) {
        panic!("PAPER_SMOKE output path must contain SMOKE");
    }
    path
}

fn require_smoke_name(smoke: bool, path: &str) -> bool {
    !smoke || path.contains("SMOKE")
}

fn push(report: &mut String, text: &str) {
    report.push_str(text);
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::{Arm, MoveLibrary, SoapProposalMode, apply_arm, campaign, require_smoke_name};

    #[test]
    fn outline_counts_use_the_registered_blocks() {
        let lj38 = campaign("lj38").unwrap();
        assert_eq!(
            (lj38.seed0, lj38.seed_count, lj38.paper_budget),
            (2000, 72, Some(400_000))
        );
        assert_eq!(lj38.reference, Some(-173.928427));
        let lj75 = campaign("lj75").unwrap();
        assert_eq!(lj75.paper_budget, Some(4_000_000));
        assert_eq!(lj75.seed_count, 48);
        let held = campaign("lj31").unwrap();
        assert_eq!((held.seed0, held.seed_count), (2200, 48));
        assert!(held.paper_budget.is_none());
        assert!(held.reference.is_none());
        assert_eq!(campaign("lj102").unwrap().reference, Some(-569.363652));
        let soap = campaign("soap38").unwrap();
        assert_eq!((soap.seed0, soap.seed_count), (5000, 64));
        assert_eq!(soap.paper_budget, Some(25_000));
        assert!(soap.icosahedral);
        assert_eq!(campaign("morse38-6").unwrap().reference, Some(-157.477108));
        assert!(campaign("morse80-6").unwrap().reference.is_none());
        assert_eq!(campaign("morse80-14").unwrap().reference, Some(-340.811371));
        assert!(campaign("lj12-groups").is_err());
    }

    #[test]
    fn arms_keep_the_named_presets() {
        let derived = apply_arm(13, Arm::Derived);
        assert!(derived.budget_window && derived.bayes_screen);
        let control = apply_arm(13, Arm::ForCluster);
        assert!(!control.budget_window && control.bayes_screen);
        let tau = anneal_core::screen::cost_asymmetric_threshold(
            control.screen_steps,
            control.relax_steps,
        );
        assert!((control.bayes_threshold - tau).abs() < 1e-12);
        assert!((tau - 7.0 / 15.0).abs() < 1e-12);
        assert_eq!(control.screen_steps, 25);
        assert_eq!(control.temperature.to_bits(), 0.8_f64.to_bits());
        let recommended = apply_arm(13, Arm::Recommended);
        let off = apply_arm(13, Arm::SoapOff);
        assert_eq!(off.soap_mode, SoapProposalMode::Off);
        assert_eq!(off.move_library, MoveLibrary::LeanBurst);
        assert_eq!(
            apply_arm(13, Arm::SoapOn).soap_mode,
            SoapProposalMode::Flexible
        );
        assert_eq!(recommended.move_library, MoveLibrary::LeanBurst);
        assert_eq!(
            apply_arm(38, Arm::Recommended).move_library,
            MoveLibrary::Reseed
        );
        assert_eq!(
            apply_arm(38, Arm::SoapOn).move_library,
            MoveLibrary::LeanBurst
        );
        assert!(matches!(
            super::parse_arm("population"),
            Some(Arm::Population)
        ));
        assert!(require_smoke_name(true, "paper-controls-SMOKE.txt"));
        assert!(!require_smoke_name(true, "paper-controls.txt"));
    }
}
