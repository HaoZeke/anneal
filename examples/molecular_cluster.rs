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
//! Engine is xtb (default), cp2k, rgpot, nwchemc, or cpmdc.

mod common;

#[cfg(feature = "graphkey")]
use anneal_core::methods::archive_search::{Archive, archive_search};
use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

use rgpot_core::profile::{ProfileEvaluation, ProfileRequest, ProfileSession};
use rgpot_core::rpc::client::RpcClient;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// The rgpot route: energy and forces over Cap'n Proto from an rgpot server,
/// which hosts whatever backend it was started with. `Metatomic:<model.pt>`
/// serves a metatomic model such as PET-MAD; the same server ABI carries the
/// quantum-chemistry backends. One charged unit per calculate call, exactly as
/// for the piped helper, and no engine code in this driver at all.
struct RgpotEngine {
    client: RpcClient,
    atmnrs: Vec<i32>,
    box_: [f64; 9],
    /// Calls the server refused.
    failures: usize,
}

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

/// The piped engine: one child process, many evaluations.
struct Engine {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    symbols: Vec<&'static str>,
    /// Evaluations the engine refused (failed SCF), reported at the end.
    failures: usize,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Close stdin so the helper's read loop sees EOF and exits after
        // finishing any in-flight write. Killing first closes the pipe
        // under that write and the helper reports BrokenPipe.
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

fn start_engine(m: usize, engine: &str) -> Engine {
    let helper = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/ase_objective.py");
    // One OpenMP team per xtb-cli call: the driver is already serial, and
    // inheriting the host's default (all cores) oversubscribes every EVAL.
    let omp = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".into());
    let mut child = Command::new("python3")
        .arg(helper)
        .env("ASE_ENGINE", engine)
        .env("PYTHONUNBUFFERED", "1")
        .env("OMP_NUM_THREADS", omp)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to start the ASE helper");
    let stdout = child.stdout.take().expect("helper stdout");
    let mut symbols = Vec::new();
    for _ in 0..m {
        symbols.extend_from_slice(&SYMBOLS);
    }
    Engine {
        child,
        reader: BufReader::new(stdout),
        symbols,
        failures: 0,
    }
}

impl Engine {
    /// One charged evaluation: energy in eV, forces to gradient in eV/A.
    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let n = x.len() / 3;
        let stdin = self.child.stdin.as_mut()?;
        let mut msg = format!("{n}\n");
        for i in 0..n {
            msg.push_str(&format!(
                "{} {:.10} {:.10} {:.10}\n",
                self.symbols[i],
                x[3 * i],
                x[3 * i + 1],
                x[3 * i + 2]
            ));
        }
        msg.push_str("EVAL\n");
        stdin.write_all(msg.as_bytes()).ok()?;
        stdin.flush().ok()?;
        let mut line = String::new();
        let nread = self.reader.read_line(&mut line).ok()?;
        if nread == 0 {
            return None;
        }
        let failed = line.starts_with("FAIL");
        let energy: f64 = if failed {
            f64::INFINITY
        } else {
            line.trim().strip_prefix("E ")?.parse().ok()?
        };
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            let mut fl = String::new();
            self.reader.read_line(&mut fl).ok()?;
            let p: Vec<f64> = fl
                .split_whitespace()
                .filter_map(|v| v.parse().ok())
                .collect();
            if p.len() == 3 {
                // Gradient is minus the force.
                for k in 0..3 {
                    g[3 * i + k] = -p[k];
                }
            }
        }
        let mut done = String::new();
        self.reader.read_line(&mut done).ok()?;
        if failed {
            self.failures += 1;
            return None;
        }
        Some((energy, g))
    }
}

/// A conforming potential backend behind rgpot's minimum in-process profile.
/// The shared loader owns one persistent session, exchanges the schema's
/// `ForceInput` and `PotentialResult` carriers, and returns eV/eV-per-Angstrom
/// values directly to this driver.
struct ProfileEngine {
    session: ProfileSession,
    atmnrs: Vec<i32>,
    failures: usize,
}

fn profile_prefix(engine: &str) -> Option<&str> {
    match engine {
        "nwchemc" | "cpmdc" => Some(engine),
        _ => None,
    }
}

fn optimizer_value_gradient(evaluation: ProfileEvaluation) -> (f64, Array1<f64>) {
    let gradient = Array1::from(
        evaluation
            .forces
            .into_iter()
            .map(|force| -force)
            .collect::<Vec<_>>(),
    );
    (evaluation.energy, gradient)
}

fn molecular_profile_request<'a>(
    positions: &'a [f64],
    atomic_numbers: &'a [i32],
) -> ProfileRequest<'a> {
    ProfileRequest {
        positions,
        atomic_numbers,
        box_matrix: None,
        length_unit: "angstrom",
        energy_unit: "eV",
    }
}

impl ProfileEngine {
    fn load(m: usize, prefix: &str) -> Self {
        let config_variable = format!("{}_CONFIG", prefix.to_ascii_uppercase());
        let config_path = std::env::var("POTENTIAL_CONFIG")
            .or_else(|_| std::env::var(&config_variable))
            .unwrap_or_else(|_| panic!("POTENTIAL_CONFIG or {config_variable}"));
        let config = std::fs::read(&config_path).expect("PotentialConfig message");
        let explicit_library = std::env::var("POTENTIAL_LIBRARY").ok();
        let session = unsafe {
            ProfileSession::load(
                prefix,
                explicit_library.as_deref().map(std::path::Path::new),
                &config,
            )
        }
        .unwrap_or_else(|error| panic!("load {prefix} profile: {error}"));
        log_line(&format!(
            "  profile {} {} ABI {} from {}",
            session.prefix(),
            session.version(),
            session.abi_version(),
            session.library_path()
        ));
        let atmnrs: Vec<i32> = (0..m).flat_map(|_| [8i32, 1, 1]).collect();
        Self {
            session,
            atmnrs,
            failures: 0,
        }
    }

    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let positions = x.iter().copied().collect::<Vec<_>>();
        let request = molecular_profile_request(&positions, &self.atmnrs);
        match self.session.evaluate(&request) {
            Ok(evaluation) => Some(optimizer_value_gradient(evaluation)),
            Err(_) => {
                self.failures += 1;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforming_embed_backends_share_profile_path() {
        assert_eq!(profile_prefix("nwchemc"), Some("nwchemc"));
        assert_eq!(profile_prefix("cpmdc"), Some("cpmdc"));
        assert_eq!(profile_prefix("rgpot"), None);
        assert_eq!(profile_prefix("xtb"), None);
    }

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
        .unwrap_or_else(|| "xtb".into());
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let n = 3 * m;
    log_line(&format!(
        "(H2O){m} through {engine}, {budget} charged evaluations, {seeds} seeds"
    ));

    let groups: Vec<Vec<usize>> = (0..m).map(|g| (3 * g..3 * g + 3).collect()).collect();
    let embed_prefix = profile_prefix(&engine);
    let mut rg_eng = if engine == "rgpot" {
        Some(RgpotEngine::connect(m))
    } else {
        None
    };
    let mut profile_eng = embed_prefix.map(|prefix| ProfileEngine::load(m, prefix));
    let mut eng = if engine == "rgpot" || embed_prefix.is_some() {
        None
    } else {
        Some(start_engine(m, &engine))
    };
    for seed in seed0..(seed0 + seeds) {
        let failures_before = rg_eng
            .as_ref()
            .map(|candidate| candidate.failures)
            .or_else(|| profile_eng.as_ref().map(|candidate| candidate.failures))
            .or_else(|| eng.as_ref().map(|candidate| candidate.failures))
            .unwrap_or(0);
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
                match (&mut rg_eng, &mut profile_eng, &mut eng) {
                    (Some(r), _, _) => r.eval(v),
                    (_, Some(profile), _) => profile.eval(v),
                    (_, _, Some(p)) => p.eval(v),
                    _ => None,
                }
            });
            (f, xr)
        };
        // Start: molecules on a loose sphere, rigid, no overlaps.
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(1));
        let mut x0 = Array1::zeros(3 * n);
        for (g, atoms) in groups.iter().enumerate() {
            let r = 3.0 + (g as f64) * 0.1;
            let th = rng.random::<f64>() * std::f64::consts::TAU;
            let ph = (rng.random::<f64>() * 2.0 - 1.0).acos();
            let c = [
                r * ph.sin() * th.cos(),
                r * ph.sin() * th.sin(),
                r * ph.cos(),
            ];
            for (a, &idx) in atoms.iter().enumerate() {
                for k in 0..3 {
                    x0[3 * idx + k] = c[k] + WATER[a][k];
                }
            }
        }
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
        let failures = rg_eng
            .as_ref()
            .map(|r| r.failures)
            .or_else(|| profile_eng.as_ref().map(|e| e.failures))
            .or_else(|| eng.as_ref().map(|p| p.failures))
            .unwrap_or(0)
            .saturating_sub(failures_before);
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
