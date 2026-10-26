//! Smoke of the recommended arm on Sutton-Chen Pt38.
//!
//! The energy and forces come from rgpot (`rgpot_sutton_chen_energy_forces`).
//! This file does not contain the Sutton-Chen formula. It is a smoke,
//! not a paper row. The reference energy -10117.2454 is the Doye and
//! Wales 10-8 Pt38 Oh value, printed for comparison. A short budget is
//! not expected to reach it.
//!
//! Usage: `RGPOT_LIB=/path/to/librgpot.so sutton_chen_pt38_smoke [budget] [seed]`

use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use anneal_core::methods::cluster_hopping::{Config, Ledger, optimize};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};

const REFERENCE: f64 = -10117.2454;

#[repr(C)]
struct ScConfig {
    epsilon: f64,
    c: f64,
    a: f64,
    n: i32,
    m: i32,
    cutoff: f64,
    periodic: i32,
}

type ForceFn =
    unsafe extern "C" fn(*const ScConfig, i32, *const f64, *const f64, *mut f64, *mut f64) -> i32;

struct RgpotSc {
    _handle: *mut c_void,
    force: ForceFn,
}

// The search moves the closure between threads and calls it from one
// of them. The C library serializes its pair list.
unsafe impl Send for RgpotSc {}
unsafe impl Sync for RgpotSc {}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

impl RgpotSc {
    fn open(path: &str) -> Self {
        let c_path = CString::new(path).expect("library path");
        let handle = unsafe { dlopen(c_path.as_ptr(), 2) };
        if handle.is_null() {
            let err = unsafe { dlerror() };
            let msg = if err.is_null() {
                "dlopen failed".to_string()
            } else {
                unsafe { CStr::from_ptr(err) }
                    .to_string_lossy()
                    .into_owned()
            };
            panic!("cannot load {path}: {msg}");
        }
        let symbol = CString::new("rgpot_sutton_chen_energy_forces").unwrap();
        let raw = unsafe { dlsym(handle, symbol.as_ptr()) };
        if raw.is_null() {
            panic!("librgpot has no rgpot_sutton_chen_energy_forces");
        }
        Self {
            _handle: handle,
            force: unsafe { std::mem::transmute::<*mut c_void, ForceFn>(raw) },
        }
    }

    /// Energy and gradient. The gradient is minus the rgpot force.
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = (x.len() / 3) as i32;
        let cfg = ScConfig {
            epsilon: 1.0,
            c: 34.408,
            a: 1.0,
            n: 10,
            m: 8,
            cutoff: 40.0,
            periodic: 0,
        };
        let box_ = [80.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 80.0];
        let mut energy = 0.0;
        let mut forces = vec![0.0; x.len()];
        let rc = unsafe {
            (self.force)(
                &cfg,
                n,
                x.as_ptr(),
                box_.as_ptr(),
                &mut energy,
                forces.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_sutton_chen_energy_forces failed");
        let gradient = Array1::from_iter(forces.iter().map(|f| -f));
        (energy, gradient)
    }
}

fn main() {
    let path = env::var("RGPOT_LIB").unwrap_or_else(|_| {
        eprintln!("set RGPOT_LIB to librgpot");
        std::process::exit(2);
    });
    let args: Vec<String> = env::args().collect();
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(800);
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);

    let potential = RgpotSc::open(&path);
    let cfg = Config::recommended(38);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (value, relaxed, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(potential.value_and_gradient(v))
            },
            |_, _| true,
        );
        (value, relaxed)
    };
    let out = optimize(&cfg, &mut ledger, &mut relax, seed);
    println!("label: smoke, not a paper row");
    println!("arm: recommended");
    println!("atoms: 38");
    println!("potential: rgpot Sutton-Chen 10-8");
    println!("reference: {REFERENCE}");
    println!("best_energy: {}", out.best);
    println!("force_calls: {}", ledger.spent());
    println!("hops: {}", out.hops);
    println!("delta_from_reference: {}", out.best - REFERENCE);
}
