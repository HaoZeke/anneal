//! In-process rgpot kernels. No potserv, no Python pipe.
//!
//! XTB: `rgpot_xtb_create` / `rgpot_xtb_force` from `libxtb_engine.so`.
//! CuH2: `rgpot_cuh2_force` from the Fortran kernel shared object.

use libloading::{Library, Symbol};
use ndarray::{Array1, ArrayView1};
use std::path::PathBuf;

const GFN2: i32 = 3;

#[repr(C)]
struct XtbConfig {
    method: i32,
    accuracy: f64,
    electronic_temperature: f64,
    max_iterations: i32,
    charge: f64,
    uhf: i32,
}

type XtbCreate =
    unsafe extern "C" fn(*const XtbConfig, *mut u8, usize) -> *mut std::ffi::c_void;
type XtbDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);
type XtbForce = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    i64,
    *const f64,
    *const i32,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
) -> i32;
type Cuh2Force = unsafe extern "C" fn(
    i32,
    *const f64,
    *const i32,
    *const f64,
    *mut f64,
    *mut f64,
) -> i32;
type FortranLastError = unsafe extern "C" fn(*mut u8, i32) -> i32;

fn first_existing(candidates: &[PathBuf]) -> PathBuf {
    for path in candidates {
        if path.as_os_str().is_empty() {
            continue;
        }
        if path.exists() {
            return path.clone();
        }
    }
    candidates
        .iter()
        .find(|path| !path.as_os_str().is_empty())
        .cloned()
        .unwrap_or_else(|| PathBuf::from("libxtb_engine.so"))
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Persistent GFN2-xTB session loaded from `libxtb_engine.so`.
pub(crate) struct XtbDirect {
    _lib: Library,
    pot: *mut std::ffi::c_void,
    force: Symbol<'static, XtbForce>,
    destroy: Symbol<'static, XtbDestroy>,
    atmnrs: Vec<i32>,
    box_: [f64; 9],
    pub failures: usize,
}

impl Drop for XtbDirect {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pot) };
    }
}

impl XtbDirect {
    pub(crate) fn load(atmnrs: Vec<i32>, box_: [f64; 9]) -> Self {
        let path = first_existing(&[
            env_path("RGPOT_XTB_ENGINE").unwrap_or_default(),
            env_path("XTB_ENGINE").unwrap_or_default(),
            PathBuf::from("libxtb_engine.so"),
        ]);
        let lib = unsafe { Library::new(&path) }
            .unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
        let create: Symbol<XtbCreate> = unsafe { lib.get(b"rgpot_xtb_create\0") }
            .expect("rgpot_xtb_create");
        let destroy: Symbol<XtbDestroy> = unsafe { lib.get(b"rgpot_xtb_destroy\0") }
            .expect("rgpot_xtb_destroy");
        let force: Symbol<XtbForce> =
            unsafe { lib.get(b"rgpot_xtb_force\0") }.expect("rgpot_xtb_force");
        let cfg = XtbConfig {
            method: GFN2,
            accuracy: 1.0,
            electronic_temperature: 300.0,
            max_iterations: 250,
            charge: 0.0,
            uhf: 0,
        };
        let mut err = [0u8; 256];
        let pot = unsafe { create(&cfg, err.as_mut_ptr(), err.len()) };
        if pot.is_null() {
            let msg = String::from_utf8_lossy(&err);
            panic!("rgpot_xtb_create failed: {msg}");
        }
        println!("  rgpot direct xtb from {}", path.display());
        // libloading symbols borrow the Library. Extend the lifetime to the
        // struct that owns `_lib` for the process.
        let force = unsafe { std::mem::transmute::<Symbol<XtbForce>, Symbol<'static, XtbForce>>(force) };
        let destroy =
            unsafe { std::mem::transmute::<Symbol<XtbDestroy>, Symbol<'static, XtbDestroy>>(destroy) };
        Self {
            _lib: lib,
            pot,
            force,
            destroy,
            atmnrs,
            box_,
            failures: 0,
        }
    }

    pub(crate) fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let n = x.len() / 3;
        let mut forces = vec![0.0; 3 * n];
        let mut energy = 0.0;
        let mut variance = 0.0;
        let pos: Vec<f64> = x.iter().copied().collect();
        let rc = unsafe {
            (self.force)(
                self.pot,
                n as i64,
                pos.as_ptr(),
                self.atmnrs.as_ptr(),
                forces.as_mut_ptr(),
                &mut energy,
                &mut variance,
                self.box_.as_ptr(),
            )
        };
        if rc != 0 || !energy.is_finite() {
            self.failures += 1;
            if self.failures == 1 || self.failures % 50 == 0 {
                eprintln!("  rgpot xtb failure {} status {rc} energy {energy}", self.failures);
            }
            return None;
        }
        let g = Array1::from(forces.into_iter().map(|f| -f).collect::<Vec<_>>());
        Some((energy, g))
    }
}

/// Persistent CuH2 EAM kernel loaded from the Fortran C ABI.
pub(crate) struct Cuh2Direct {
    _lib: Library,
    force: Symbol<'static, Cuh2Force>,
    last_error: Option<Symbol<'static, FortranLastError>>,
    atmnrs: Vec<i32>,
    box_: [f64; 9],
    pub failures: usize,
}

impl Cuh2Direct {
    pub(crate) fn load(atmnrs: Vec<i32>, box_: [f64; 9]) -> Self {
        let path = first_existing(&[
            env_path("RGPOT_CUH2_LIBRARY").unwrap_or_default(),
            PathBuf::from("librgpot_cuh2.so"),
        ]);
        let lib = unsafe { Library::new(&path) }
            .unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
        let force: Symbol<Cuh2Force> = unsafe { lib.get(b"rgpot_cuh2_force\0") }
            .expect("rgpot_cuh2_force");
        let force =
            unsafe { std::mem::transmute::<Symbol<Cuh2Force>, Symbol<'static, Cuh2Force>>(force) };
        let last_error = unsafe { lib.get(b"rgpot_fortran_last_error\0") }.ok().map(
            |symbol: Symbol<FortranLastError>| {
                unsafe {
                    std::mem::transmute::<Symbol<FortranLastError>, Symbol<'static, FortranLastError>>(
                        symbol,
                    )
                }
            },
        );
        println!("  rgpot direct cuh2 from {}", path.display());
        Self {
            _lib: lib,
            force,
            last_error,
            atmnrs,
            box_,
            failures: 0,
        }
    }

    pub(crate) fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let n = x.len() / 3;
        let mut forces = vec![0.0; 3 * n];
        let mut energy = 0.0;
        let pos: Vec<f64> = x.iter().copied().collect();
        let rc = unsafe {
            (self.force)(
                n as i32,
                pos.as_ptr(),
                self.atmnrs.as_ptr(),
                self.box_.as_ptr(),
                forces.as_mut_ptr(),
                &mut energy,
            )
        };
        if rc != 0 || !energy.is_finite() {
            self.failures += 1;
            if self.failures == 1 || self.failures % 50 == 0 {
                let mut buf = [0u8; 256];
                let nwrite = self.last_error.as_ref().map_or(0, |err| unsafe {
                    (err)(buf.as_mut_ptr(), buf.len() as i32)
                });
                let msg = if nwrite > 0 {
                    String::from_utf8_lossy(&buf[..nwrite as usize]).into_owned()
                } else {
                    String::new()
                };
                eprintln!(
                    "  rgpot cuh2 failure {} status {rc} energy {energy} {msg}",
                    self.failures
                );
            }
            return None;
        }
        let g = Array1::from(forces.into_iter().map(|f| -f).collect::<Vec<_>>());
        Some((energy, g))
    }
}
