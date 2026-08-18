//! (H2O)m search. The potential is an rgpot handle viewed as an eindir
//! objective. Anneal only runs [`cluster_search`].
//!
//! Usage: molecular_cluster <m_molecules> <budget> <seeds>

mod common;

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, SoapProposalMode, repack_rigid_groups,
};
use anneal_core::methods::cluster_search::{search_from_maybe_bank, verify};
use common::efficiency::{bank_label, report_trace};
use common::rgpot_eindir::RgpotObjective;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;

const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.7572, 0.5865, 0.0],
    [-0.7572, 0.5865, 0.0],
];

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(400);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let n = 3 * m;
    let atmnrs: Vec<i32> = (0..m).flat_map(|_| [8i32, 1, 1]).collect();
    let species: Vec<u32> = (0..m).flat_map(|_| [8, 1, 1]).collect();
    let groups: Vec<Vec<usize>> = (0..m).map(|g| (3 * g..3 * g + 3).collect()).collect();
    let pot = RgpotObjective::xtb(&atmnrs, [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0]);
    let obj = pot.wrapper();
    let mut cfg = Config::recommended_molecular(species, groups.clone(), 1.0);
    cfg.move_library = MoveLibrary::Molecular {
        groups: groups.clone(),
        reactive: false,
    };
    cfg.soap_mode = soap_mode_from_env();
    write_resolved_config(&cfg);
    println!(
        "(H2O){m} through eindir/rgpot xtb, arm {}, budget {budget}, seeds {seed0}..{}",
        bank_label(),
        seed0 + seeds
    );
    for seed in seed0..seed0 + seeds {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(1));
        let mut template = Array1::zeros(3 * n);
        for atoms in &groups {
            for (a, &idx) in atoms.iter().enumerate() {
                for k in 0..3 {
                    template[3 * idx + k] = WATER[a][k];
                }
            }
        }
        let x0 = repack_rigid_groups(template.view(), &groups, cfg.length_scale, &mut rng);
        let mut ledger = Ledger::new(budget);
        let (out, stats) = search_from_maybe_bank(&obj, &cfg, &mut ledger, x0.view(), seed);
        let checked = verify(&obj, &out);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  charged {}  converged {}/{}  arm {}",
            out.best,
            out.hops,
            ledger.spent(),
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
            println!("    verify e={e:.6} |g|={g:.3e}");
        }
        report_trace(&out, ledger.spent());
        if let Some(bx) = out.best_state {
            let path = format!("best_h2o{m}_eindir_s{seed}.xyz");
            let mut f = std::fs::File::create(&path).expect("xyz");
            writeln!(f, "{n}\nbest {:.6} eV", out.best).ok();
            for i in 0..n {
                writeln!(
                    f,
                    "{} {:.6} {:.6} {:.6}",
                    if i % 3 == 0 { "O" } else { "H" },
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
