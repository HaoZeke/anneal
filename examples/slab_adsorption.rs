//! Adsorbate search on a substrate. The potential is an rgpot handle
//! viewed as an eindir objective. Anneal only runs [`cluster_search`].
//!
//! Usage: slab_adsorption <con_file> <budget> <seeds> [plain|recommended]

mod common;

use anneal_core::methods::cluster_hopping::{Config, Ledger, SoapProposalMode, covalent_radius};
use anneal_core::methods::cluster_search::{search_from_maybe_bank, verify};
use common::efficiency::{apply_two_phase, bank_label, report_eval_wall, report_trace};
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use common::slab::{Mobile, adsorbate_groups, displace_adsorbate, read_system, search_arm, symbol};
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
    let groups = adsorbate_groups(&species, &free_seeds);
    let mut cfg = if search == "plain" {
        Config::for_molecular(species.clone(), groups, 1.0)
    } else {
        Config::recommended_molecular(species.clone(), groups, 1.0)
    };
    cfg.active_region = Some((free_seeds.clone(), 1));
    if !free_seeds.is_empty() && free_seeds.len() < n {
        let adsorbate = free_seeds
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
    for seed in seed0..seed0 + seeds {
        let x0 = displace_adsorbate(&base_x, &species, &free_seeds, box_, seed);
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
