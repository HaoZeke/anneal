//! Molecular-cluster search against an external quantum-chemistry engine.
//!
//! The beyond-Lennard-Jones demonstration: (H2O)m under GFN2-xTB or PBE
//! through CP2K, with the objective behind a persistent piped helper
//! (`examples/ase_objective.py`) so engine startup is paid once. The move
//! library is the molecular one: every arm rigid on the declared groups, the
//! same shake / relocate / burst vocabulary whose atomic form carried the
//! measured Lennard-Jones results. The ledger charges one unit per
//! energy-and-forces evaluation, which at density-functional prices is the
//! only honest unit there is.
//!
//! Usage: molecular_cluster <m_molecules> <budget> <seeds> [engine]
//! Engine is xtb (default) or cp2k, forwarded as ASE_ENGINE.

use anneal_core::methods::cluster_hopping::{run_with_gradient, Config, Ledger};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// One rigid water template, in Angstrom.
const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.7572, 0.5865, 0.0],
    [-0.7572, 0.5865, 0.0],
];
const SYMBOLS: [&str; 3] = ["O", "H", "H"];

/// The piped engine: one child process, many evaluations.
struct Engine {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    symbols: Vec<&'static str>,
    /// Evaluations the engine refused (failed SCF), reported at the end.
    failures: usize,
}

fn start_engine(m: usize, engine: &str) -> Engine {
    let helper = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/ase_objective.py");
    let mut child = Command::new("python3")
        .arg(helper)
        .env("ASE_ENGINE", engine)
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
        self.reader.read_line(&mut line).ok()?;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let engine = args.get(4).cloned().unwrap_or_else(|| "xtb".into());
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let n = 3 * m;
    println!("(H2O){m} through {engine}, {budget} charged evaluations, {seeds} seeds");

    let groups: Vec<Vec<usize>> = (0..m).map(|g| (3 * g..3 * g + 3).collect()).collect();
    for seed in seed0..(seed0 + seeds) {
        let mut eng = start_engine(m, &engine);
        let mut ledger = Ledger::new(budget);
        // The recommended stack's allocator over the molecular library.
        let mut cfg = Config::recommended(n);
        cfg.burst_moves = false;
        cfg.temperature = 0.8;
        // Water oxygen-oxygen contacts sit near 2.8 A; the group contact
        // cutoff has to see them.
        cfg.molecular_groups = Some(groups.clone());
        cfg.group_cutoff = 3.4;
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
                eng.eval(v)
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
            let c = [r * ph.sin() * th.cos(), r * ph.sin() * th.sin(), r * ph.cos()];
            for (a, &idx) in atoms.iter().enumerate() {
                for k in 0..3 {
                    x0[3 * idx + k] = c[k] + WATER[a][k];
                }
            }
        }
        let out = run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  scf failures {}",
            out.best, out.hops, eng.failures
        );
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
            println!("  wrote {path}");
        }
        let _ = eng.child.kill();
    }
}
