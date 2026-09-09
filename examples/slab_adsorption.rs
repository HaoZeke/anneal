//! Adsorbate search on a substrate. The potential is an rgpot handle
//! viewed as an eindir objective. Anneal only runs [`cluster_search`].
//!
//! Usage: slab_adsorption <con_file> <budget> <seeds> [plain|recommended]

mod common;

use anneal_core::methods::cluster_hopping::{Config, Ledger, SoapProposalMode, covalent_radius};
use anneal_core::methods::cluster_search::{search_from_maybe_bank, verify};
use common::efficiency::{apply_two_phase, bank_label, report_eval_wall, report_trace};
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use common::slab::{
    Mobile, adsorbate_groups, hop_atoms, place_adsorbates, read_system, search_arm, symbol,
};
use std::io::Write;

fn soap_mode_from_env() -> SoapProposalMode {
    match std::env::var("ANNEAL_SOAP_MODE").as_deref() {
        Ok("flexible") | Err(_) => SoapProposalMode::Flexible,
        Ok("rigid") => SoapProposalMode::Rigid,
        Ok("off") => SoapProposalMode::Off,
        Ok(value) => panic!("ANNEAL_SOAP_MODE must be flexible, rigid, or off; got {value}"),
    }
}

fn write_resolved_config(cfg: &Config) {
    let Ok(path) = std::env::var("ANNEAL_RESOLVED_CONFIG") else {
        return;
    };
    std::fs::write(
        &path,
        cfg.resolved_json()
            .expect("serialize resolved configuration"),
    )
    .unwrap_or_else(|error| panic!("write resolved configuration {path}: {error}"));
}

fn arm_from_args(args: &[String]) -> &'static str {
    match args.get(4).map(String::as_str) {
        Some("plain") => "plain",
        Some("recommended") => "recommended",
        Some(value) => panic!("search arm must be plain or recommended; got {value}"),
        None => search_arm(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con = args
        .get(1)
        .cloned()
        .expect("usage: slab_adsorption <con_file> <budget> <seeds> [plain|recommended]");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(25);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let search = arm_from_args(&args);
    let (base_x, species, free_seeds, box_) = read_system(&con);
    let n = species.len();
    let atmnrs: Vec<i32> = species.iter().map(|&z| z as i32).collect();
    let hop = hop_atoms(&free_seeds);
    let groups = adsorbate_groups(&base_x, &species, &hop);
    let mut cfg = if search == "plain" {
        Config::for_molecular(species.clone(), groups, 1.0)
    } else {
        Config::recommended_molecular(species.clone(), groups, 1.0)
    };
    // Hop the adsorbate only. shells=0 so the first Cu shell is not
    // added to the move mask; the quench still relaxes free substrate
    // through Mobile.
    cfg.active_region = Some((hop.clone(), 0));
    if !hop.is_empty() && hop.len() < n {
        let adsorbate = hop
            .iter()
            .map(|&i| covalent_radius(species[i]))
            .fold(0.0_f64, f64::max);
        cfg.length_scale = 2.0 * adsorbate;
    }
    cfg.screen_steps = 10;
    cfg.relax_steps = 150;
    if !free_seeds.is_empty() && free_seeds.len() < n {
        let mut frozen = vec![true; n];
        for &i in &free_seeds {
            frozen[i] = false;
        }
        cfg.frozen = Some(frozen);
    }
    cfg.soap_mode = soap_mode_from_env();
    apply_two_phase(&mut cfg);
    write_resolved_config(&cfg);
    let pot = RgpotObjective::cuh2(&atmnrs, box_);
    emit_engine_manifest("cuh2");
    let inner = pot.wrapper();
    let mut active = vec![false; n];
    for &i in &free_seeds {
        active[i] = true;
    }
    let obj = Mobile {
        inner: &inner,
        active,
    };
    println!(
        "{con}: {n} atoms through eindir/rgpot cuh2, {} free, search {search}, arm {}, budget {budget}, seeds {seed0}..{}",
        free_seeds.len(),
        bank_label(),
        seed0 + seeds
    );
    if let Some(replicas) = std::env::var("HISTORY_REPLICAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        drop(obj);
        drop(inner);
        run_slab_ensembles(
            &cfg,
            budget,
            seed0,
            seeds,
            replicas,
            &atmnrs,
            &species,
            &free_seeds,
            &base_x,
            box_,
        );
        return;
    }
    for seed in seed0..seed0 + seeds {
        let x0 = place_adsorbates(&base_x, &species, &free_seeds, box_, seed);
        if seed == seed0 {
            report_eval_wall(&obj, x0.view(), "cuh2");
        }
        let mut ledger = Ledger::new(budget);
        let (out, stats) = search_from_maybe_bank(&obj, &cfg, &mut ledger, x0.view(), seed);
        let checked = verify(&obj, &out);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  charged {}  basins {}  converged {}/{}  arm {}  search {search}",
            out.best,
            out.hops,
            ledger.spent(),
            out.basins,
            stats.converged,
            stats.total(),
            bank_label()
        );
        println!(
            "    quench charged screen/full/check {}/{}/{}  screens {}  capped {}",
            stats.screen_charged,
            stats.full_charged,
            stats.check_charged,
            stats.screens,
            stats.capped
        );
        if let Some((e, g)) = checked {
            println!("    verify e={e:.6} |g|_mobile={g:.3e}");
        }
        report_trace(&out, ledger.spent());
        if let Some(bx) = out.best_state {
            let path = format!("best_slab_eindir_s{seed}.xyz");
            let mut f = std::fs::File::create(&path).expect("xyz");
            writeln!(f, "{n}\nbest {:.6} eV", out.best).ok();
            for i in 0..n {
                writeln!(
                    f,
                    "{} {:.6} {:.6} {:.6}",
                    symbol(species[i]),
                    bx[3 * i],
                    bx[3 * i + 1],
                    bx[3 * i + 2]
                )
                .ok();
            }
            println!("  wrote {path}");
        }
    }
}

/// Thread replicas of the surface search under the ensemble contract, one
/// rgpot engine per replica; only the mobile atoms carry a gradient.
#[allow(clippy::too_many_arguments)]
fn run_slab_ensembles(
    cfg: &Config,
    budget: usize,
    seed0: u64,
    seeds: u64,
    replicas: usize,
    atmnrs: &[i32],
    species: &[u32],
    free_seeds: &[usize],
    base_x: &ndarray::Array1<f64>,
    box_: [f64; 9],
) {
    use anneal_core::methods::ensemble::{
        EnsembleProblem, ObjectiveFactory, StartFactory, run_ensemble,
    };
    use anneal_core::methods::minima_hopping::SerializedWitness;
    use anneal_core::pes_exploration::StructureContext;
    use common::ensemble_report::{
        Tally, config_from_env, print_header, print_report, print_tally,
    };
    use eindir_core::gradient::DifferentiableObjective;
    use ndarray::{Array1, ArrayView1};
    use rand::Rng;
    use std::sync::Mutex;

    let n = species.len();
    let ens = config_from_env(replicas, budget, None);
    ens.validate(cfg).unwrap_or_else(|error| panic!("{error}"));
    let descriptor = anneal_core::catalog::molecular::descriptor_space(species)
        .unwrap_or_else(|error| panic!("slab descriptor space: {error:?}"));
    let context = StructureContext::new(Some(species.to_vec()), None, Some("slab-cuh2".into()));
    let radius = 1e-4;
    #[cfg(feature = "ira")]
    let witness = SerializedWitness(Mutex::new(
        anneal_core::shape::IraStructureWitness {
            kmax_factor: 1.8,
            radius,
        }
        .with_pair_cache(128 * 1024 * 1024),
    ));
    #[cfg(not(feature = "ira"))]
    let witness = {
        use anneal_core::bias::{Fingerprint, SortedPairs};
        let fingerprint = SortedPairs { n_points: n };
        SerializedWitness(Mutex::new(
            move |left: ArrayView1<f64>, right: ArrayView1<f64>| {
                let l = fingerprint.describe(left);
                let r = fingerprint.describe(right);
                l.iter()
                    .zip(r.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
                    < radius
            },
        ))
    };
    let mut active = vec![false; n];
    for &i in free_seeds {
        active[i] = true;
    }
    let objective: ObjectiveFactory<'_> = &|_| {
        let pot = RgpotObjective::cuh2(atmnrs, box_);
        let active = active.clone();
        Box::new(move |x: ArrayView1<f64>| {
            let inner = pot.wrapper();
            Mobile {
                inner: &inner,
                active: active.clone(),
            }
            .value_and_gradient(x)
        })
    };
    let start: StartFactory<'_> =
        &|_, rng| place_adsorbates(base_x, species, free_seeds, box_, rng.random::<u64>());
    let same_family = |_: &[f64], _: &[f64]| false;
    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        // The engine's gradient noise (GFN2 at accuracy 0.01) sits far
        // above the LJ share bound, so the record criterion itself is the
        // certificate here and the polish window runs one decade above it.
        certificate: cfg.record_gradient,
        polish_below: 10.0 * cfg.record_gradient,
        callbacks_per_objective: 1,
    };
    print_header(
        &ens,
        if cfg!(feature = "ira") {
            "ira-cached"
        } else {
            "sorted-pairs-fallback"
        },
        "slab-cuh2",
        &format!("record gradient {:.2e},", cfg.record_gradient),
    );
    let audit = RgpotObjective::cuh2(atmnrs, box_);
    let audit_active = active.clone();
    let verify = |replica: usize, x: &Array1<f64>, reported: f64| {
        let inner = audit.wrapper();
        let (e, g) = Mobile {
            inner: &inner,
            active: audit_active.clone(),
        }
        .value_and_gradient(x.view());
        let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            e.is_finite() && (e - reported).abs() < 1e-5,
            "replica {replica} reports {reported} but its coordinates have energy {e}"
        );
        (e, gmax)
    };
    let mut tally = Tally::new();
    for seed in seed0..seed0 + seeds {
        let report = run_ensemble(cfg, &ens, seed, &problem)
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        print_report(seed, &ens, &report, " eV", &verify, &mut tally);
    }
    print_tally(&ens, &tally, None);
}
