//! Build an eindir objective from an rgpot potential.
//!
//! Anneal does not evaluate XTB or CuH2. rgpot owns the kernel; eindir
//! owns the objective surface; this file only constructs the IS-A handle
//! the search already knows how to take.

#![allow(dead_code)]

use anneal_core::catalog::FreshEvaluation;
use anneal_core::catalog::molecular::{GFN2_ACCURACY, GFN2_MAX_ITERATIONS};
use anneal_core::compatibility::{AbiStamp, EngineDescriptor, ProtocolVersion};
use eindir_core::ffi::{EindirObjectiveWrapper, eindir_core_abi_compatible, eindir_objective_t};
use eindir_core::gradient::DifferentiableObjective;
use libloading::{Library, Symbol};
use rgpot_core::eindir::{
    rgpot_eindir_abi_stamp, rgpot_potential_free_eindir, rgpot_potential_new_eindir,
};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{rgpot_tensor_data, rgpot_tensor_owned_cpu_f64_2d};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};
use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

const GFN2: i32 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_maps_the_eindir_stamp_and_optional_build_identity() {
        let stamp = eindir_core::ffi::eindir_abi_stamp_t {
            abi_major: 1,
            abi_minor: 2,
            objective_layout: 3,
            objective_size: 64,
            objective_align: 8,
            dlpack_major: 1,
            dlpack_minor: 4,
            features: 0b101,
        };
        let descriptor = descriptor_from_stamp("xtb", stamp, Some("rgpot@abc123"));
        assert_eq!(descriptor.engine_id, "rgpot.xtb");
        assert_eq!(descriptor.protocol_family, "rgpot.potentials");
        assert_eq!(descriptor.protocol, ProtocolVersion::new(1, 0));
        assert_eq!(descriptor.abi.layout_revision, 3);
        assert_eq!(descriptor.abi.objective_size, 64);
        assert_eq!(descriptor.abi.objective_align, 8);
        assert_eq!(descriptor.abi.dlpack_minor, 4);
        assert_eq!(descriptor.abi.features, 0b101);
        assert_eq!(descriptor.build_identity.as_deref(), Some("rgpot@abc123"));
    }
}

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
    evaluations: usize,
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

fn descriptor_from_stamp(
    backend: &str,
    stamp: eindir_core::ffi::eindir_abi_stamp_t,
    build_identity: Option<&str>,
) -> EngineDescriptor {
    let mut descriptor = EngineDescriptor::with_family(
        format!("rgpot.{backend}"),
        "rgpot.potentials",
        ProtocolVersion::new(1, 0),
        AbiStamp::from_eindir_stamp(stamp),
    );
    descriptor.build_identity = build_identity
        .filter(|identity| !identity.is_empty())
        .map(str::to_owned);
    descriptor
}

fn engine_descriptor(backend: &str) -> EngineDescriptor {
    let build_identity = std::env::var("RGPOT_BUILD_IDENTITY").ok();
    descriptor_from_stamp(backend, rgpot_eindir_abi_stamp(), build_identity.as_deref())
}

pub(crate) fn emit_engine_manifest(backend: &str) {
    let descriptor = engine_descriptor(backend);
    println!(
        "engine_manifest={}",
        serde_json::to_string(&descriptor).expect("engine descriptor must serialize")
    );
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
    let kernel = unsafe { &mut *(user as *mut XtbKernel) };
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
    kernel.evaluations += 1;
    if let Some(path) = env_path("RGPOT_XTB_TRACE")
        && let Ok(mut trace) = File::create(path)
    {
        let positions = unsafe { std::slice::from_raw_parts(pos, 3 * n) };
        let atomic_numbers = unsafe { std::slice::from_raw_parts(z, n) };
        let _ = writeln!(trace, "{n}");
        let _ = writeln!(trace, "rgpot xtb evaluation {}", kernel.evaluations);
        for i in 0..n {
            let _ = writeln!(
                trace,
                "{} {:.17e} {:.17e} {:.17e}",
                atomic_numbers[i],
                positions[3 * i],
                positions[3 * i + 1],
                positions[3 * i + 2]
            );
        }
    }
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
    let rc = unsafe { (kernel.force)(n as i32, pos, z, boxp, forces.as_mut_ptr(), &mut energy) };
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
    dim: usize,
}

// SAFETY: the handle is uniquely owned and rgpot declares its embedded
// potential Send. Callers serialize evaluation because the handle is not Sync.
unsafe impl Send for RgpotObjective {}

impl Drop for RgpotObjective {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free_eindir(self.pot) };
    }
}

impl RgpotObjective {
    pub(crate) fn wrapper(&self) -> EindirObjectiveWrapper<'_> {
        let stamp = rgpot_eindir_abi_stamp();
        assert_eq!(
            unsafe { eindir_core_abi_compatible(&stamp) },
            1,
            "rgpot/eindir ABI stamp is incompatible with anneal's eindir boundary"
        );
        let obj = unsafe { &*(self.pot as *const eindir_objective_t) };
        unsafe { EindirObjectiveWrapper::new(obj) }
    }

    pub(crate) fn fresh_evaluation(&self, coordinates: &[f64]) -> Result<FreshEvaluation, String> {
        if coordinates.len() != self.dim || coordinates.iter().any(|value| !value.is_finite()) {
            return Err("rgpot coordinate dimension or values are invalid".into());
        }
        let objective = self.wrapper();
        let (energy, gradient) =
            objective.value_and_gradient(ndarray::ArrayView1::from(coordinates));
        if !energy.is_finite()
            || gradient.len() != self.dim
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err("rgpot returned an invalid energy or gradient".into());
        }
        Ok(FreshEvaluation {
            energy,
            forces: gradient.iter().map(|value| -*value).collect(),
        })
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
            accuracy: GFN2_ACCURACY,
            electronic_temperature: 300.0,
            max_iterations: GFN2_MAX_ITERATIONS,
            charge: 0.0,
            uhf: 0,
        };
        let mut err = [0u8; 256];
        let pot = unsafe { create(&cfg, err.as_mut_ptr(), err.len()) };
        if pot.is_null() {
            panic!("rgpot_xtb_create failed: {}", String::from_utf8_lossy(&err));
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
            evaluations: 0,
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
        let _ = std::io::Write::flush(&mut std::io::stdout());
        Self {
            pot: handle,
            dim: 3 * n,
        }
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
        let _ = std::io::Write::flush(&mut std::io::stdout());
        Self {
            pot: handle,
            dim: 3 * n,
        }
    }
}
