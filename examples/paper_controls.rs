#![allow(dead_code, unused_imports)]
use anneal_core::catalog::lj::system_signature;
#[cfg(test)]
use anneal_core::methods::cluster_hopping::MoveLibrary;
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, SoapProposalMode, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::run_manifest::RunManifest;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
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
        let (outcome, wall) = run_seed(&cfg, &campaign, seed, budget);
        let force_to_hit = campaign.reference.and_then(|reference| {
            outcome
                .improvements
                .iter()
                .find(|(_, _, _, energy)| *energy <= reference + HIT)
                .map(|(_, spent, _, _)| *spent)
        });
        let hit = match (campaign.reference, force_to_hit) {
            (None, _) => "unscored",
            (Some(_), Some(_)) => "true",
            (Some(_), None) => "false",
        };
        push(
            &mut report,
            &format!(
                "seed: {seed}\nforce_count: {}\nforce_to_hit: {}\nwall_seconds: {wall:.6}\n\
                 best: {:.6}\nhit: {hit}\n",
                outcome.charged,
                force_to_hit
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".into()),
                outcome.best,
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
        _ => return None,
    })
}

fn apply_arm(n: usize, arm: Arm) -> Config {
    match arm {
        Arm::Recommended | Arm::SoapOn => Config::recommended(n),
        Arm::ForCluster => Config::for_cluster(n),
        Arm::Derived => Config::derived(n),
        Arm::SoapOff => {
            let mut cfg = Config::recommended(n);
            cfg.soap_mode = SoapProposalMode::Off;
            cfg
        }
    }
}

fn run_seed(
    cfg: &Config,
    campaign: &Campaign,
    seed: u64,
    budget: usize,
) -> (anneal_core::methods::cluster_hopping::Outcome, f64) {
    let kind = match campaign.rho {
        Some(rho) => PairKind::Morse { rho },
        None => PairKind::LennardJones,
    };
    let potential = match kind {
        PairKind::LennardJones => PairPotential::lennard_jones(campaign.n),
        PairKind::Morse { rho } => PairPotential::morse(campaign.n, rho),
    };
    let start = if campaign.icosahedral {
        ico_start(campaign.n, seed)
    } else {
        let mut start_rng = StdRng::seed_from_u64(seed);
        random_cluster_in_radius(
            campaign.n,
            cfg.start_radius(),
            cfg.min_separation,
            &mut start_rng,
        )
    };
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
    let started = Instant::now();
    let outcome = run_with_gradient(cfg, start.view(), &mut ledger, &mut relax, None, &mut rng);
    (outcome, started.elapsed().as_secs_f64())
}

fn ico_start(n: usize, _seed: u64) -> Array1<f64> {
    Array1::zeros(n * 3)
}

fn lj_reference(_n: usize) -> Option<f64> {
    None
}

fn morse_reference(_n: usize, _rho: f64) -> Option<f64> {
    None
}

fn engine_identity(_n: usize, _rho: Option<f64>) -> (String, String) {
    (String::new(), String::new())
}

fn output_path(smoke: bool) -> String {
    if smoke { "SMOKE".into() } else { "out".into() }
}

fn push(report: &mut String, text: &str) {
    report.push_str(text);
}

fn git_commit() -> String {
    "unknown".into()
}

fn cpu_model() -> String {
    "unknown".into()
}
