//! Build an eindir objective from an rgpot potential.
//!
//! Anneal does not evaluate XTB or CuH2. rgpot owns the kernel; eindir
//! owns the objective surface; this file only constructs the IS-A handle
//! the search already knows how to take.

use eindir_core::ffi::{EindirObjectiveWrapper, eindir_objective_t};
use libloading::{Library, Symbol};
use rgpot_core::eindir::{rgpot_potential_free_eindir, rgpot_potential_new_eindir};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_owned_cpu_f64_2d};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};
use std::ffi::c_void;
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

type XtbCreate = unsafe extern "C" fn(*const XtbConfig, *mut u8, usize) -> *mut c_void;
type XtbDestroy = unsafe extern "C" fn(*mut c_void);
type XtbForce = unsafe extern "C" fn(
    *mut c_void,
    i64,
    *const f64,
    *const i32,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
) -> i32;
type Cuh2Force =
    unsafe extern "C" fn(i32, *const f64, *const i32, *const f64, *mut f64, *mut f64) -> i32;

struct XtbKernel {
    _lib: Library,
    pot: *mut c_void,
    force: Symbol<'static, XtbForce>,
    destroy: Symbol<'static, XtbDestroy>,
}

impl Drop for XtbKernel {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pot) };
    }
}

struct Cuh2Kernel {
    _lib: Library,
    force: Symbol<'static, Cuh2Force>,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn first_existing(cands: &[PathBuf]) -> PathBuf {
    cands
        .iter()
        .find(|p| !p.as_os_str().is_empty() && p.exists())
        .cloned()
        .unwrap_or_else(|| cands[cands.len() - 1].clone())
}

unsafe extern "C" fn xtb_callback(
    user: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let kernel = unsafe { &*(user as *const XtbKernel) };
    let inp = unsafe { &*input };
    let out = unsafe { &mut *output };
    let n = match unsafe { inp.n_atoms() } {
        Some(n) => n,
        None => return rgpot_status_t::RGPOT_INVALID_PARAMETER,
    };
    let pos = unsafe { rgpot_tensor_data(inp.positions) } as *const f64;
    let z = unsafe { rgpot_tensor_data(inp.atomic_numbers) } as *const i32;
    let boxp = unsafe { rgpot_tensor_data(inp.box_matrix) } as *const f64;
    if pos.is_null() || z.is_null() || boxp.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let mut forces = vec![0.0; 3 * n];
    let mut energy = 0.0;
    let mut var = 0.0;
    let rc = unsafe {
        (kernel.force)(
            kernel.pot,
            n as i64,
            pos,
            z,
            forces.as_mut_ptr(),
            &mut energy,
            &mut var,
            boxp,
        )
    };
    if rc != 0 || !energy.is_finite() {
        return rgpot_status_t::RGPOT_INTERNAL_ERROR;
    }
    out.energy = energy;
    out.variance = var;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), n as i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn cuh2_callback(
    user: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let kernel = unsafe { &*(user as *const Cuh2Kernel) };
    let inp = unsafe { &*input };
    let out = unsafe { &mut *output };
    let n = match unsafe { inp.n_atoms() } {
        Some(n) => n,
        None => return rgpot_status_t::RGPOT_INVALID_PARAMETER,
    };
    let pos = unsafe { rgpot_tensor_data(inp.positions) } as *const f64;
    let z = unsafe { rgpot_tensor_data(inp.atomic_numbers) } as *const i32;
    let boxp = unsafe { rgpot_tensor_data(inp.box_matrix) } as *const f64;
    if pos.is_null() || z.is_null() || boxp.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let mut forces = vec![0.0; 3 * n];
    let mut energy = 0.0;
    let rc = unsafe {
        (kernel.force)(
            n as i32,
            pos,
            z,
            boxp,
            forces.as_mut_ptr(),
            &mut energy,
        )
    };
    if rc != 0 || !energy.is_finite() {
        return rgpot_status_t::RGPOT_INTERNAL_ERROR;
    }
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), n as i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn drop_xtb(p: *mut c_void) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p as *mut XtbKernel) });
    }
}

unsafe extern "C" fn drop_cuh2(p: *mut c_void) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p as *mut Cuh2Kernel) });
    }
}

/// An rgpot potential viewed as an eindir objective.
pub(crate) struct RgpotObjective {
    pot: *mut rgpot_core::eindir::rgpot_potential_t,
}

impl Drop for RgpotObjective {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free_eindir(self.pot) };
    }
}

impl RgpotObjective {
    pub(crate) fn wrapper(&self) -> EindirObjectiveWrapper<'_> {
        let obj = unsafe { &*(self.pot as *const eindir_objective_t) };
        unsafe { EindirObjectiveWrapper::new(obj) }
    }

    pub(crate) fn xtb(atmnrs: &[i32], box_: [f64; 9]) -> Self {
        let path = first_existing(&[
            env_path("RGPOT_XTB_ENGINE").unwrap_or_default(),
            PathBuf::from("libxtb_engine.so"),
        ]);
        let lib = unsafe { Library::new(&path) }
            .unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
        let create: Symbol<XtbCreate> =
            unsafe { lib.get(b"rgpot_xtb_create\0") }.expect("rgpot_xtb_create");
        let destroy: Symbol<XtbDestroy> =
            unsafe { lib.get(b"rgpot_xtb_destroy\0") }.expect("rgpot_xtb_destroy");
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
            panic!(
                "rgpot_xtb_create failed: {}",
                String::from_utf8_lossy(&err)
            );
        }
        let force =
            unsafe { std::mem::transmute::<Symbol<XtbForce>, Symbol<'static, XtbForce>>(force) };
        let destroy = unsafe {
            std::mem::transmute::<Symbol<XtbDestroy>, Symbol<'static, XtbDestroy>>(destroy)
        };
        let kernel = Box::new(XtbKernel {
            _lib: lib,
            pot,
            force,
            destroy,
        });
        let n = atmnrs.len();
        let low = vec![-80.0; 3 * n];
        let high = vec![80.0; 3 * n];
        let handle = unsafe {
            rgpot_potential_new_eindir(
                xtb_callback,
                Box::into_raw(kernel) as *mut c_void,
                Some(drop_xtb),
                n,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                low.as_ptr(),
                high.as_ptr(),
            )
        };
        assert!(!handle.is_null(), "rgpot_potential_new_eindir xtb");
        println!("  eindir objective from rgpot xtb ({})", path.display());
        Self { pot: handle }
    }

    pub(crate) fn cuh2(atmnrs: &[i32], box_: [f64; 9]) -> Self {
        let path = first_existing(&[
            env_path("RGPOT_CUH2_LIBRARY").unwrap_or_default(),
            PathBuf::from("librgpot_cuh2.so"),
        ]);
        let lib = unsafe { Library::new(&path) }
            .unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
        let force: Symbol<Cuh2Force> =
            unsafe { lib.get(b"rgpot_cuh2_force\0") }.expect("rgpot_cuh2_force");
        let force =
            unsafe { std::mem::transmute::<Symbol<Cuh2Force>, Symbol<'static, Cuh2Force>>(force) };
        let kernel = Box::new(Cuh2Kernel { _lib: lib, force });
        let n = atmnrs.len();
        let low = vec![-80.0; 3 * n];
        let high = vec![80.0; 3 * n];
        let handle = unsafe {
            rgpot_potential_new_eindir(
                cuh2_callback,
                Box::into_raw(kernel) as *mut c_void,
                Some(drop_cuh2),
                n,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                low.as_ptr(),
                high.as_ptr(),
            )
        };
        assert!(!handle.is_null(), "rgpot_potential_new_eindir cuh2");
        println!("  eindir objective from rgpot cuh2 ({})", path.display());
        Self { pot: handle }
    }
}
