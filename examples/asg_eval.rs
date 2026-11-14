// Copyright (c) anneal developers
//! Persistent evaluator for the ASE and GOFEE panel beyond Lennard-Jones.
//!
//! ```text
//! asg_eval spec <system>
//! asg_eval start <system> <seed>
//! asg_eval serve <system>
//! ```
//!
//! `spec` prints the atom count, dimension, paper budget, reference energy
//! and the length the GOFEE box is scaled from. `start` prints the same
//! initial geometry `h2h_metal`, `h2h_covalent` and `h2h_seed` draw for that
//! seed. `serve` reads one geometry per stdin line and writes the energy,
//! the seconds spent inside the potential, and the gradient. Gupta is
//! `Gupta::copper_gold`. Stillinger-Weber and Tersoff come from
//! `librgpot_covalent.so` with the same scaling as `h2h_covalent`. TIP4P is
//! `GminTip4p` on the rigid `6 n` chart.

use anneal_core::methods::cluster_hopping::{Config, random_cluster_in_radius};
use anneal_core::potentials::{GminTip4p, Gupta, random_tip4p_cluster};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::ffi::{CStr, CString, c_char, c_double, c_int, c_long, c_void};
use std::io::{BufRead, Write};
use std::time::Instant;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;

type EvalFn =
    unsafe extern "C" fn(c_int, c_long, *const c_double, *mut c_double, *mut c_double) -> c_int;
type UnitsFn = unsafe extern "C" fn(*mut c_double, *mut c_double);

struct Covalent {
    _handle: *mut c_void,
    eval: EvalFn,
    kind: c_int,
    length: f64,
    energy: f64,
}

impl Covalent {
    fn open(kind: c_int) -> Self {
        let path = std::env::var("RGPOT_COVALENT_LIB")
            .unwrap_or_else(|_| "/work/covalent-h2h/lib/librgpot_covalent.so".to_string());
        let c_path = CString::new(path.clone()).expect("library path");
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            let err = unsafe { dlerror() };
            let msg = if err.is_null() {
                "dlopen failed".to_string()
            } else {
                unsafe { CStr::from_ptr(err) }
                    .to_string_lossy()
                    .into_owned()
            };
            panic!("rgpot covalent library {path}: {msg}");
        }
        let eval = load_fn::<EvalFn>(handle, "rgpot_covalent_energy_gradient");
        let (length, energy) = if kind == 0 {
            let units = load_fn::<UnitsFn>(handle, "rgpot_covalent_sw_units");
            let (mut sigma, mut epsilon) = (0.0, 0.0);
            unsafe { units(&mut sigma, &mut epsilon) };
            (sigma, epsilon)
        } else {
            (1.0, 1.0)
        };
        Self {
            _handle: handle,
            eval,
            kind,
            length,
            energy,
        }
    }

    fn evaluate(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let positions: Vec<f64> = x.iter().map(|value| value * self.length).collect();
        let mut energy = 0.0;
        let mut gradient = vec![0.0; x.len()];
        let status = unsafe {
            (self.eval)(
                self.kind,
                n as c_long,
                positions.as_ptr(),
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        if status != 0 || !energy.is_finite() {
            return (f64::INFINITY, Array1::zeros(x.len()));
        }
        let scale = self.length / self.energy;
        (
            energy / self.energy,
            Array1::from_vec(gradient.into_iter().map(|g| g * scale).collect()),
        )
    }

    fn dimer(&self, r: f64) -> (f64, f64) {
        let (energy, gradient) =
            self.evaluate(Array1::from_vec(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]).view());
        (energy, gradient[3])
    }
}

fn load_fn<T>(handle: *mut c_void, name: &str) -> T {
    let c_name = CString::new(name).expect("symbol");
    unsafe {
        dlerror();
        let symbol = dlsym(handle, c_name.as_ptr());
        if symbol.is_null() {
            let err = dlerror();
            let msg = if err.is_null() {
                "missing symbol".to_string()
            } else {
                CStr::from_ptr(err).to_string_lossy().into_owned()
            };
            panic!("{name}: {msg}");
        }
        std::mem::transmute_copy(&symbol)
    }
}

enum Model {
    Gupta(Gupta),
    Water(GminTip4p),
    Covalent(Covalent),
}

struct Spec {
    n_atoms: usize,
    dim: usize,
    budget: usize,
    reference: f64,
    length: f64,
    symbols: Vec<&'static str>,
    model: Model,
}

fn gupta_length(n_copper: usize, n_gold: usize) -> f64 {
    let (copper, gold) = if n_gold == 0 {
        (2, 0)
    } else if n_copper == 0 {
        (0, 2)
    } else {
        (1, 1)
    };
    let dimer = Gupta::copper_gold(copper, gold);
    let mut best_r = 0.0;
    let mut best_e = f64::INFINITY;
    let mut r = 1.8;
    while r <= 4.5 {
        let mut x = Array1::zeros(6);
        x[3] = r;
        let energy = dimer.value_and_gradient(x.view()).0;
        if energy < best_e {
            best_e = energy;
            best_r = r;
        }
        r += 0.002;
    }
    // Same reduced length the covalent dimer probe reports: the pair
    // minimum divided by 2^(1/6), so a GOFEE scale of 3.4 / length puts
    // that minimum on the Angstrom site the repulsive prior was written for.
    best_r / 2.0_f64.powf(1.0 / 6.0)
}

fn symbols_repeat(symbol: &'static str, count: usize) -> Vec<&'static str> {
    vec![symbol; count]
}

fn spec_of(name: &str) -> Spec {
    match name {
        "au55" => metal(0, 55, 1_000_000, -191.03085280),
        "cu38" => metal(38, 0, 400_000, -108.96759055),
        "cu12au26" => metal(12, 26, 400_000, -127.21076866),
        "sw10" => covalent(10, 0, 100_000, -13.79658026, "Si"),
        "tc10" => covalent(10, 1, 100_000, -55.08979464, "C"),
        "tip4p8" => {
            let n = 8;
            Spec {
                n_atoms: 2 * n,
                dim: 6 * n,
                budget: 2_000_000,
                reference: -305.5183221676,
                // The oxygen diameter is already on the Angstrom scale the
                // GOFEE prior uses, so the rigid chart is not rescaled.
                length: 3.4,
                // A label for the generator. The potential reads the rigid
                // chart, not the element.
                symbols: symbols_repeat("Ar", 2 * n),
                model: Model::Water(GminTip4p::new(n)),
            }
        }
        other => panic!("unknown system {other}"),
    }
}

fn metal(n_copper: usize, n_gold: usize, budget: usize, reference: f64) -> Spec {
    let n = n_copper + n_gold;
    let mut symbols = symbols_repeat("Cu", n_copper);
    symbols.extend(symbols_repeat("Au", n_gold));
    Spec {
        n_atoms: n,
        dim: 3 * n,
        budget,
        reference,
        length: gupta_length(n_copper, n_gold),
        symbols,
        model: Model::Gupta(Gupta::copper_gold(n_copper, n_gold)),
    }
}

fn covalent(n: usize, kind: c_int, budget: usize, reference: f64, symbol: &'static str) -> Spec {
    let model = Covalent::open(kind);
    let (length, _) = dimer_scales(&model);
    Spec {
        n_atoms: n,
        dim: 3 * n,
        budget,
        reference,
        length,
        symbols: symbols_repeat(symbol, n),
        model: Model::Covalent(model),
    }
}

/// Bond length over 2^(1/6), and the well depth. Same probe as `h2h_covalent`.
fn dimer_scales(potential: &Covalent) -> (f64, f64) {
    let mut r = 0.05_f64;
    let mut before = potential.dimer(r);
    let (mut lo, mut hi, mut slope_lo, mut slope_hi) = loop {
        let next = 1.2 * r;
        if next >= 1.0e3 {
            panic!("dimer probe did not find a well");
        }
        let now = potential.dimer(next);
        if before.1 < 0.0 && now.1 > 0.0 && before.0.min(now.0) < 0.0 {
            break (r, next, before.1, now.1);
        }
        (r, before) = (next, now);
    };
    let mut kept = 0_i32;
    while hi - lo > 1.0e-10 * hi {
        let mid = (lo * slope_hi - hi * slope_lo) / (slope_hi - slope_lo);
        let (_, slope) = potential.dimer(mid);
        if slope < 0.0 {
            lo = mid;
            slope_lo = slope;
            kept = kept.min(0) - 1;
            if kept <= -2 {
                slope_hi *= 0.5;
            }
        } else if slope > 0.0 {
            hi = mid;
            slope_hi = slope;
            kept = kept.max(0) + 1;
            if kept >= 2 {
                slope_lo *= 0.5;
            }
        } else {
            lo = mid;
            hi = mid;
        }
    }
    let bond = 0.5 * (lo + hi);
    let (energy, _) = potential.dimer(bond);
    assert!(energy < 0.0, "dimer energy {energy} is not a well");
    (bond / 2.0_f64.powf(1.0 / 6.0), -energy)
}

fn value_and_gradient(spec: &Spec, x: ArrayView1<f64>) -> (f64, Array1<f64>, f64) {
    let mark = Instant::now();
    let (energy, gradient) = match &spec.model {
        Model::Gupta(potential) => potential.value_and_gradient(x),
        Model::Water(potential) => potential.value_and_gradient(x),
        Model::Covalent(potential) => potential.evaluate(x),
    };
    (energy, gradient, mark.elapsed().as_secs_f64())
}

fn start_of(spec: &Spec, name: &str, seed: u64) -> Array1<f64> {
    match &spec.model {
        Model::Gupta(_) => {
            let (n_copper, n_gold) = match name {
                "au55" => (0, 55),
                "cu38" => (38, 0),
                "cu12au26" => (12, 26),
                other => panic!("not a Gupta system {other}"),
            };
            let n = n_copper + n_gold;
            let species = Gupta::copper_gold_species(n_copper, n_gold);
            let cfg = Config::recommended(n).with_species(species, None);
            let mut rng = StdRng::seed_from_u64(seed);
            random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
        }
        Model::Covalent(potential) => {
            let (length, energy) = dimer_scales(potential);
            let cfg = Config::with_scales(spec.n_atoms, length, energy);
            let mut rng = StdRng::seed_from_u64(seed);
            random_cluster_in_radius(
                spec.n_atoms,
                cfg.start_radius(),
                cfg.min_separation,
                &mut rng,
            )
        }
        Model::Water(_) => {
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
            random_tip4p_cluster(8, &mut rng)
        }
    }
}

fn write_xyz(coords: &Array1<f64>) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for atom in coords.exact_chunks(3) {
        writeln!(out, "{:.10} {:.10} {:.10}", atom[0], atom[1], atom[2]).unwrap();
    }
}

fn cmd_spec(name: &str) {
    let spec = spec_of(name);
    println!(
        "{} {} {} {} {:.12} {:.12}",
        name, spec.n_atoms, spec.dim, spec.budget, spec.reference, spec.length
    );
    println!("{}", spec.symbols.join(" "));
}

fn cmd_start(name: &str, seed: u64) {
    let spec = spec_of(name);
    let coords = start_of(&spec, name, seed);
    assert_eq!(coords.len(), spec.dim);
    write_xyz(&coords);
}

fn cmd_serve(name: &str) {
    let spec = spec_of(name);
    let stdin = std::io::stdin();
    let mut lines = stdin.lock();
    let mut line = String::new();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    loop {
        line.clear();
        if lines.read_line(&mut line).expect("stdin") == 0 {
            break;
        }
        let text = line.trim();
        if text.is_empty() || text == "quit" {
            break;
        }
        let mut coords = Array1::zeros(spec.dim);
        let mut count = 0;
        for word in text.split_whitespace() {
            if count >= spec.dim {
                panic!("serve got more than {} numbers", spec.dim);
            }
            coords[count] = word.parse::<f64>().expect("coordinate");
            count += 1;
        }
        if count != spec.dim {
            panic!("serve got {count} numbers, expected {}", spec.dim);
        }
        let (energy, gradient, inside) = value_and_gradient(&spec, coords.view());
        write!(out, "{energy:.16e} {inside:.9e}").unwrap();
        for value in gradient.iter() {
            write!(out, " {value:.16e}").unwrap();
        }
        writeln!(out).unwrap();
        out.flush().unwrap();
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    match cmd {
        "spec" => cmd_spec(args.get(1).map(String::as_str).unwrap_or("")),
        "start" => {
            let name = args.get(1).map(String::as_str).unwrap_or("");
            let seed = args
                .get(2)
                .and_then(|text| text.parse().ok())
                .expect("seed");
            cmd_start(name, seed);
        }
        "serve" => cmd_serve(args.get(1).map(String::as_str).unwrap_or("")),
        _ => {
            eprintln!("usage: asg_eval spec|start|serve SYSTEM");
            std::process::exit(2);
        }
    }
}
