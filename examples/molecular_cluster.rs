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

use anneal_core::methods::cluster_hopping::{
    run_with_gradient, Config, Ledger, MoveLibrary,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

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


/// The in-process NWChem engine: dlopen of the split libnwchemc C ABI.
/// One session per process; positions in per call as plain arrays, energy
/// and forces back in Hartree and Hartree/Bohr, converted here to eV and
/// eV/Angstrom. NWCHEMC_LIBRARY names the shared object, NWCHEMC_PARAMS
/// a serialized NWChemParams message (capnp encode output); without the
/// blob the schema defaults apply through an empty message.
struct NwchemcEngine {
    _lib: libloading::Library,
    session: *mut std::ffi::c_void,
    calc: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        i32,
        *const f64,
        *const i32,
        *mut f64,
    ) -> NWChemCResult,
    atmnrs: Vec<i32>,
    failures: usize,
}

#[repr(C)]
struct NWChemCResult {
    ok: i32,
    energy_h: f64,
    message: [u8; 512],
}

const HARTREE_EV: f64 = 27.211386245988;
const BOHR_ANG: f64 = 0.529177210903;

impl NwchemcEngine {
    fn load(m: usize) -> Self {
        let path = std::env::var("NWCHEMC_LIBRARY").expect("NWCHEMC_LIBRARY");
        let lib = unsafe { libloading::Library::new(&path) }.expect("dlopen libnwchemc");
        let params = match std::env::var("NWCHEMC_PARAMS") {
            Ok(p) => std::fs::read(p).expect("params blob"),
            // A minimal flat message: one empty segment table entry and an
            // all-default root struct, which the reader treats as defaults.
            Err(_) => vec![0u8; 8],
        };
        let session = unsafe {
            let create: libloading::Symbol<
                unsafe extern "C" fn(*const std::ffi::c_void, usize) -> *mut std::ffi::c_void,
            > = lib.get(b"nwchemc_session_create").expect("session_create");
            create(params.as_ptr() as *const _, params.len())
        };
        assert!(!session.is_null(), "nwchemc session creation failed");
        let calc = unsafe {
            let s: libloading::Symbol<
                unsafe extern "C" fn(
                    *mut std::ffi::c_void,
                    i32,
                    *const f64,
                    *const i32,
                    *mut f64,
                ) -> NWChemCResult,
            > = lib.get(b"nwchemc_session_energy_forces").expect("energy_forces");
            *s
        };
        let atmnrs: Vec<i32> = (0..m).flat_map(|_| [8i32, 1, 1]).collect();
        Self { _lib: lib, session, calc, atmnrs, failures: 0 }
    }

    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let na = self.atmnrs.len();
        let pos: Vec<f64> = x.iter().cloned().collect();
        let mut forces = vec![0f64; 3 * na];
        let res = unsafe {
            (self.calc)(
                self.session,
                na as i32,
                pos.as_ptr(),
                self.atmnrs.as_ptr(),
                forces.as_mut_ptr(),
            )
        };
        if res.ok == 0 {
            self.failures += 1;
            return None;
        }
        let g = Array1::from(
            forces
                .iter()
                .map(|f| -f * HARTREE_EV / BOHR_ANG)
                .collect::<Vec<f64>>(),
        );
        Some((res.energy_h * HARTREE_EV, g))
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
        let mut rg_eng = if engine == "rgpot" {
            Some(RgpotEngine::connect(m))
        } else {
            None
        };
        let mut nw_eng = if engine == "nwchemc" {
            Some(NwchemcEngine::load(m))
        } else {
            None
        };
        let mut eng = if engine == "rgpot" || engine == "nwchemc" {
            None
        } else {
            Some(start_engine(m, &engine))
        };
        let mut ledger = Ledger::new(budget);
        let species: Vec<u32> = (0..m).flat_map(|_| [8, 1, 1]).collect();
        let energy_scale = std::env::var("ENERGY_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        // The recommended stack's allocator over the molecular library.
        let mut cfg = Config::recommended_molecular(
            species,
            groups.clone(),
            energy_scale,
        );
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
                match (&mut rg_eng, &mut nw_eng, &mut eng) {
                    (Some(r), _, _) => r.eval(v),
                    (_, Some(nwe), _) => nwe.eval(v),
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
            let c = [r * ph.sin() * th.cos(), r * ph.sin() * th.sin(), r * ph.cos()];
            for (a, &idx) in atoms.iter().enumerate() {
                for k in 0..3 {
                    x0[3 * idx + k] = c[k] + WATER[a][k];
                }
            }
        }
        let out = run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng);
        let failures = rg_eng
            .as_ref()
            .map(|r| r.failures)
            .or_else(|| nw_eng.as_ref().map(|e| e.failures))
            .or_else(|| eng.as_ref().map(|p| p.failures))
            .unwrap_or(0);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  engine failures {}",
            out.best, out.hops, failures
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
        if let Some(p) = eng.as_mut() {
            let _ = p.child.kill();
        }
    }
}
