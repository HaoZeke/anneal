//! (H2O)m search. The potential is an rgpot handle viewed as an eindir
//! objective. Anneal only runs [`cluster_search`].
//!
//! Usage: molecular_cluster <m_molecules> <budget> <seeds>

mod common;

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, SoapProposalMode, repack_rigid_groups,
};
use anneal_core::methods::cluster_search::{search_from_maybe_bank, verify};
use common::efficiency::{apply_two_phase, bank_label, report_eval_wall, report_trace};
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use ndarray::{Array1, ArrayView1};
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
    let seed0: u64 = anneal_core::env::parsed("SEED_OFFSET").unwrap_or(0);
    let n = 3 * m;
    let atmnrs: Vec<i32> = (0..m).flat_map(|_| [8i32, 1, 1]).collect();
    let species: Vec<u32> = (0..m).flat_map(|_| [8, 1, 1]).collect();
    let groups: Vec<Vec<usize>> = (0..m).map(|g| (3 * g..3 * g + 3).collect()).collect();
    let pot = RgpotObjective::xtb(&atmnrs, [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0]);
    emit_engine_manifest("xtb");
    let obj = pot.wrapper();
    let mut cfg = Config::recommended_molecular(species.clone(), groups.clone(), 1.0);
    cfg.move_library = MoveLibrary::Molecular {
        groups: groups.clone(),
        reactive: false,
    };
    cfg.soap_mode = soap_mode_from_env();
    apply_two_phase(&mut cfg);
    write_resolved_config(&cfg);
    println!(
        "(H2O){m} through eindir/rgpot xtb, arm {}, budget {budget}, seeds {seed0}..{}",
        bank_label(),
        seed0 + seeds
    );
    if let Some(replicas) = anneal_core::env::parsed::<usize>("HISTORY_REPLICAS") {
        drop(obj);
        run_water_ensembles(
            &cfg, m, budget, seed0, seeds, replicas, &atmnrs, &species, &groups,
        );
        return;
    }
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
        if seed == seed0 {
            report_eval_wall(&obj, x0.view(), "gfn2");
        }
        let mut ledger = Ledger::new(budget);
        let (out, stats) = search_from_maybe_bank(&obj, &cfg, &mut ledger, x0.view(), seed);
        let checked = verify(&obj, &out);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  charged {}  basins {}  converged {}/{}  arm {}",
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

/// Rigid water template placed by the group repacker.
fn water_template(n: usize, groups: &[Vec<usize>]) -> Array1<f64> {
    let mut template = Array1::zeros(3 * n);
    for atoms in groups {
        for (a, &idx) in atoms.iter().enumerate() {
            for k in 0..3 {
                template[3 * idx + k] = WATER[a][k];
            }
        }
    }
    template
}

/// Thread replicas of the water search under the ensemble contract, with
/// one xtb engine per replica (the handle is `Send`, not `Sync`).
#[allow(clippy::too_many_arguments)]
fn run_water_ensembles(
    cfg: &Config,
    m: usize,
    budget: usize,
    seed0: u64,
    seeds: u64,
    replicas: usize,
    atmnrs: &[i32],
    species: &[u32],
    groups: &[Vec<usize>],
) {
    use anneal_core::methods::cluster_hopping::repack_rigid_groups;
    use anneal_core::methods::ensemble::{EnsembleProblem, ObjectiveFactory, StartFactory};
    #[cfg(feature = "ira")]
    use anneal_core::methods::minima_hopping::SerializedWitness;
    use anneal_core::pes_exploration::StructureContext;
    use common::ensemble_report::{config_from_env, print_header, run_seeds, sorted_pairs_witness};
    use eindir_core::gradient::DifferentiableObjective;
    #[cfg(feature = "ira")]
    use std::sync::Mutex;

    let n = 3 * m;
    let ens = config_from_env(replicas, budget, None);
    ens.validate(cfg).unwrap_or_else(|error| panic!("{error}"));
    let descriptor = anneal_core::catalog::molecular::descriptor_space(species)
        .unwrap_or_else(|error| panic!("water descriptor space: {error:?}"));
    let context = StructureContext::new(
        Some(species.to_vec()),
        None,
        Some(format!("water-gfn2-m{m}")),
    );
    // Exact identity in angstrom; the molecular calibration radius.
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
    let witness = sorted_pairs_witness(n, radius);
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let objective: ObjectiveFactory<'_> = &|_| {
        let pot = RgpotObjective::xtb(atmnrs, box_);
        Box::new(move |x: ArrayView1<f64>| pot.wrapper().value_and_gradient(x))
    };
    let template = water_template(n, groups);
    let start: StartFactory<'_> =
        &|_, rng| repack_rigid_groups(template.view(), groups, cfg.length_scale, rng);
    // Packing families are a cluster notion; no two-choice crowding here.
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
        "water-gfn2",
        &format!("record gradient {:.2e},", cfg.record_gradient),
    );
    let audit = RgpotObjective::xtb(atmnrs, box_);
    let verify = |replica: usize, x: &Array1<f64>, reported: f64| {
        let (e, g) = audit.wrapper().value_and_gradient(x.view());
        let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            e.is_finite() && (e - reported).abs() < 1e-5,
            "replica {replica} reports {reported} but its coordinates have energy {e}"
        );
        (e, gmax)
    };
    run_seeds(cfg, &ens, seed0, seeds, &problem, " eV", &verify, None);
}
