//! (H2O)m search. The potential is an rgpot handle viewed as an eindir
//! objective. Anneal only runs [`cluster_search`].
//!
//! Usage: molecular_cluster <m_molecules> <budget> <seeds>

mod common;

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, repack_rigid_groups,
};
use anneal_core::methods::cluster_search::{search_from, verify};
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
    let pot = RgpotObjective::xtb(
        &atmnrs,
        [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0],
    );
    let obj = pot.wrapper();
    let mut cfg = Config::recommended_molecular(species, groups.clone(), 1.0);
    cfg.move_library = MoveLibrary::Molecular {
        groups: groups.clone(),
        reactive: false,
    };
    cfg.screen_steps = 6;
    cfg.relax_steps = 60;
    println!(
        "(H2O){m} through eindir/rgpot xtb, budget {budget}, seeds {seed0}..{}",
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
        let (out, stats) = search_from(&obj, &cfg, &mut ledger, x0.view(), seed);
        let checked = verify(&obj, &out);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  charged {}  converged {}/{}{}",
            out.best,
            out.hops,
            ledger.spent(),
            stats.converged,
            stats.total(),
            checked
                .map(|(e, g)| format!("  verify e={e:.6} |g|={g:.3e}"))
                .unwrap_or_default()
        );
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
