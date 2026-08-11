//! Molecular-cluster search against an external quantum-chemistry engine.
//!
//! The beyond-Lennard-Jones demonstration: (H2O)m through an in-process
//! minimum-profile backend (`nwchemc` or `cpmdc`), an rgpot RPC server, or a
//! persistent ASE helper for GFN2-xTB and CP2K. Every route starts once and
//! serves the full walk. The move library is the molecular one: every arm
//! rigid on the declared groups, with the same shake / relocate / burst
//! vocabulary whose atomic form carries the Lennard-Jones results. The ledger
//! charges one unit per fused energy-and-forces evaluation.
//!
//! Usage: molecular_cluster <m_molecules> <budget> <seeds> [engine]
//! Engine is rgpot (default: Cap'n Proto to potserv), nwchemc, cpmdc,
//! xtb / xtb-cli, or cp2k.

mod common;

#[cfg(feature = "graphkey")]
use anneal_core::methods::archive_search::{Archive, archive_search};
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, repack_rigid_groups, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use common::pipe_engine::PipeEngine;
#[cfg(feature = "rgpot-ex")]
use common::profile_engine::{ProfileEngine, optimizer_value_gradient, profile_prefix};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;

#[cfg(feature = "rgpot-ex")]
use rgpot_core::rpc::client::RpcClient;
#[cfg(feature = "rgpot-ex")]
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
#[cfg(feature = "rgpot-ex")]
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// The rgpot route: energy and forces over Cap'n Proto from an rgpot server,
/// which hosts whatever backend it was started with. `Metatomic:<model.pt>`
/// serves a metatomic model such as PET-MAD; the same server ABI carries the
/// quantum-chemistry backends. One charged unit per calculate call, exactly as
/// for the piped helper, and no engine code in this driver at all.
#[cfg(feature = "rgpot-ex")]
struct RgpotEngine {
    client: RpcClient,
    atmnrs: Vec<i32>,
    box_: [f64; 9],
    /// Calls the server refused.
    failures: usize,
}

#[cfg(feature = "rgpot-ex")]
impl RgpotEngine {
    fn connect(m: usize) -> Self {
        let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port: u16 = std::env::var("RGPOT_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9999);
        let mut atmnrs = Vec::new();
        for _ in 0..m {
            atmnrs.extend_from_slice(&[8, 1, 1]);
        }
        Self {
            client: RpcClient::new(&host, port).expect("rgpot client"),
            atmnrs,
            box_: [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0],
            failures: 0,
        }
    }

    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let n = x.len() / 3;
        let mut pos: Vec<f64> = x.iter().cloned().collect();
        let input = unsafe {
            rgpot_force_input_t {
                positions: rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), n as i64, 3),
                atomic_numbers: rgpot_tensor_cpu_i32_1d(self.atmnrs.as_mut_ptr(), n as i64),
                box_matrix: rgpot_tensor_cpu_f64_matrix3(self.box_.as_mut_ptr()),
            }
        };
        let mut out = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let res = self.client.calculate(&input, &mut out);
        unsafe {
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }
        match res {
            Ok(()) => {
                let mut g = Array1::zeros(3 * n);
                if !out.forces.is_null() {
                    let data = unsafe { rgpot_tensor_data(out.forces) } as *const f64;
                    if !data.is_null() {
                        for i in 0..3 * n {
                            g[i] = -unsafe { *data.add(i) };
                        }
                    }
                    unsafe { rgpot_tensor_free(out.forces) };
                }
                Some((out.energy, g))
            }
            Err(_) => {
                self.failures += 1;
                None
            }
        }
    }
}

/// One rigid water template, in Angstrom.
const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.7572, 0.5865, 0.0],
    [-0.7572, 0.5865, 0.0],
];
const SYMBOLS: [&str; 3] = ["O", "H", "H"];

fn log_line(msg: &str) {
    let mut out = std::io::stdout();
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_path_is_the_default_without_capnp() {
        assert_eq!(SYMBOLS, ["O", "H", "H"]);
    }

    #[cfg(feature = "rgpot-ex")]
    #[test]
    fn conforming_embed_backends_share_profile_path() {
        assert_eq!(profile_prefix("nwchemc"), Some("nwchemc"));
        assert_eq!(profile_prefix("cpmdc"), Some("cpmdc"));
        assert_eq!(profile_prefix("rgpot"), None);
        assert_eq!(profile_prefix("xtb"), None);
    }

    #[cfg(feature = "rgpot-ex")]
    #[test]
    fn profile_result_is_one_value_gradient_pair() {
        let evaluation = rgpot_core::profile::ProfileEvaluation {
            energy: -12.75,
            forces: vec![1.0, -2.0, 3.0],
        };

        let (energy, gradient) = optimizer_value_gradient(evaluation);

        assert_eq!(energy, -12.75);
        assert_eq!(gradient.as_slice().unwrap(), [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn molecular_profile_request_has_no_simulation_cell() {
        let positions = [0.0, 0.0, 0.0];
        let atomic_numbers = [8];
        let request = common::profile_engine::profile_request(&positions, &atomic_numbers, None);

        assert!(request.box_matrix.is_none());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let tail: Vec<String> = args.iter().skip(4).cloned().collect();
    let ras = tail.iter().any(|t| t == "ras" || t == "pair");
    let pair = tail.iter().any(|t| t == "pair");
    let engine = tail
        .iter()
        .find(|t| *t != "ras" && *t != "pair")
        .cloned()
        .unwrap_or_else(|| "rgpot".into());
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let n = 3 * m;
    log_line(&format!(
        "(H2O){m} through {engine}, {budget} charged evaluations, {seeds} seeds"
    ));

    let groups: Vec<Vec<usize>> = (0..m).map(|g| (3 * g..3 * g + 3).collect()).collect();
    #[cfg(feature = "rgpot-ex")]
    let embed_prefix = profile_prefix(&engine);
    #[cfg(not(feature = "rgpot-ex"))]
    let embed_prefix: Option<&str> = None;
    #[cfg(feature = "rgpot-ex")]
    let mut rg_eng = if engine == "rgpot" {
        Some(RgpotEngine::connect(m))
    } else {
        None
    };
    #[cfg(not(feature = "rgpot-ex"))]
    let mut rg_eng: Option<()> = if engine == "rgpot" {
        panic!("rebuild with --features rgpot-ex for the rgpot engine");
    } else {
        None
    };
    let profile_atomic_numbers: Vec<i32> = (0..m).flat_map(|_| [8i32, 1, 1]).collect();
    #[cfg(feature = "rgpot-ex")]
    let mut profile_eng =
        embed_prefix.map(|prefix| ProfileEngine::load(prefix, profile_atomic_numbers, None));
    #[cfg(not(feature = "rgpot-ex"))]
    let mut profile_eng: Option<()> = None;
    let _ = (embed_prefix, &profile_atomic_numbers);
    let mut pipe_eng = if engine == "rgpot" || embed_prefix.is_some() {
        None
    } else {
        let mut symbols = Vec::new();
        for _ in 0..m {
            symbols.extend(SYMBOLS.iter().map(|s| (*s).to_string()));
        }
        Some(PipeEngine::start(&engine, symbols, None))
    };
    for seed in seed0..(seed0 + seeds) {
        let failures_before = pipe_eng.as_ref().map(|e| e.failures).unwrap_or(0)
            + {
                #[cfg(feature = "rgpot-ex")]
                {
                    rg_eng.as_ref().map(|e| e.failures).unwrap_or(0)
                        + profile_eng.as_ref().map(ProfileEngine::failures).unwrap_or(0)
                }
                #[cfg(not(feature = "rgpot-ex"))]
                {
                    0
                }
            };
        let mut ledger = Ledger::new(budget);
        let species: Vec<u32> = (0..m).flat_map(|_| [8, 1, 1]).collect();
        let energy_scale = std::env::var("ENERGY_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        // The recommended stack's allocator over the molecular library.
        let mut cfg = Config::recommended_molecular(species, groups.clone(), energy_scale);
        let reactive = std::env::var("REACTIVE").map(|v| v == "1").unwrap_or(false);
        cfg.move_library = MoveLibrary::Molecular {
            groups: groups.clone(),
            reactive,
        };
        // Species drive the bond-matrix connectivity: groups follow the
        // structure's own bonding each hop, so a reactive event regroups the
        // moves instead of stranding the walker.
        // Screen at DFT prices: a handful of relaxation steps decides.
        cfg.screen_steps = 6;
        cfg.relax_steps = 60;

        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                if let Some(p) = pipe_eng.as_mut() {
                    return p.eval(v);
                }
                #[cfg(feature = "rgpot-ex")]
                {
                    if let Some(r) = rg_eng.as_mut() {
                        return r.eval(v);
                    }
                    if let Some(profile) = profile_eng.as_mut() {
                        return profile.eval(v);
                    }
                }
                None
            });
            (f, xr)
        };
        // Start: molecules on a loose sphere, rigid, no overlaps.
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
        #[cfg(feature = "graphkey")]
        let mut rng_ras = rng.clone();
        #[cfg(feature = "graphkey")]
        if pair {
            let mut rng_rec = rng.clone();
            let mut ledger_rec = Ledger::new(budget);
            let rec = run_with_gradient(
                &cfg,
                x0.view(),
                &mut ledger_rec,
                &mut relax,
                None,
                &mut rng_rec,
            );
            let rec_at = rec
                .improvements
                .iter()
                .rev()
                .find(|(_, _, _, e)| (*e - rec.best).abs() < 1e-8)
                .map(|(_, sp, _, _)| *sp)
                .unwrap_or(ledger_rec.spent());
            log_line(&format!(
                "  seed {seed} rec: best {:.6} eV  charged {}  hops {}  hit_at {}",
                rec.best,
                ledger_rec.spent(),
                rec.hops,
                rec_at
            ));
        }
        #[cfg(feature = "graphkey")]
        let out = if ras {
            let mut archive = Archive::new();
            let a = archive_search(
                &cfg,
                x0.view(),
                &mut ledger,
                &mut relax,
                None,
                &mut archive,
                &mut rng_ras,
            );
            if pair {
                log_line(&format!(
                    "  seed {seed} ras: best {:.6} eV  charged {}  hit_at {}  floors {} returned {} same_floor {}",
                    a.best, a.charged, a.best_at, a.floors, a.returned, a.same_floor
                ));
            }
            anneal_core::methods::cluster_hopping::Outcome {
                best: a.best,
                best_state: a.best_state,
                hops: a.full,
                ..Default::default()
            }
        } else {
            run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng)
        };
        #[cfg(not(feature = "graphkey"))]
        let out = run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng);
        let failures_after = pipe_eng.as_ref().map(|e| e.failures).unwrap_or(0)
            + {
                #[cfg(feature = "rgpot-ex")]
                {
                    rg_eng.as_ref().map(|e| e.failures).unwrap_or(0)
                        + profile_eng.as_ref().map(ProfileEngine::failures).unwrap_or(0)
                }
                #[cfg(not(feature = "rgpot-ex"))]
                {
                    0
                }
            };
        let failures = failures_after.saturating_sub(failures_before);
        log_line(&format!(
            "  seed {seed}: best {:.6} eV  hops {}  engine failures {}",
            out.best, out.hops, failures
        ));
        if let Some(bx) = out.best_state {
            let path = format!("best_h2o{m}_{engine}_s{seed}.xyz");
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
            log_line(&format!("  wrote {path}"));
        }
    }
}
