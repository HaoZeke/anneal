//! The objective gradient behind the GFN2 examples against the engine energy.
//!
//! The engine tests load the rgpot `libxtb_engine.so` named by
//! `RGPOT_XTB_ENGINE` and compare against a central difference of that
//! engine's own energy. Without it they return early. The stub test builds a
//! library with the same C ABI and a pair energy and reruns the objective
//! tests on it in a child, so the callback sign is checked without xtb.
#![cfg(all(unix, feature = "rgpot-ex"))]

#[path = "../examples/common/rgpot_eindir.rs"]
mod rgpot_eindir;

use eindir_core::gradient::DifferentiableObjective;
use eindir_core::objective::Objective;
use libloading::{Library, Symbol};
use ndarray::{Array1, ArrayView1};
use rgpot_eindir::RgpotObjective;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

const BOX: [f64; 9] = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
const STEP_A: f64 = 1e-3;
/// eV/Å. A sign error is twice the gradient, several eV/Å on these geometries.
const TOLERANCE: f64 = 2e-3;

/// A water dimer with the donor hydrogen pulled 0.12 Å along its bond and
/// the acceptor rotated off its minimum.
fn displaced_dimer() -> (Vec<i32>, Vec<f64>) {
    (
        vec![8, 1, 1, 8, 1, 1],
        vec![
            -1.551007, -0.114520, 0.000000, //
            -1.934259, 0.762503, 0.000000, //
            -0.470000, 0.040000, 0.000000, //
            1.350625, 0.111469, 0.000000, //
            1.680398, -0.373741, -0.758561, //
            1.580398, -0.523741, 0.858561,
        ],
    )
}

/// A cyclic water trimer, about 2.8 Å O–O.
fn trimer() -> (Vec<i32>, Vec<f64>) {
    (
        vec![8, 1, 1, 8, 1, 1, 8, 1, 1],
        vec![
            1.616, 0.000, 0.000, //
            0.708, -0.524, 0.030, //
            2.106, -0.236, 0.790, //
            -0.808, 1.400, 0.000, //
            -0.354, 0.613, -0.330, //
            -1.302, 1.551, 0.810, //
            -0.808, -1.400, 0.000, //
            -0.054, -1.950, 0.250, //
            -1.420, -2.000, -0.420,
        ],
    )
}

fn central_difference(energy: &mut dyn FnMut(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let mut plus = x.to_vec();
            let mut minus = x.to_vec();
            plus[i] += STEP_A;
            minus[i] -= STEP_A;
            (energy(&plus) - energy(&minus)) / (2.0 * STEP_A)
        })
        .collect()
}

fn max_residual(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn norm(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn engine_path() -> Option<PathBuf> {
    let path = std::env::var_os("RGPOT_XTB_ENGINE").filter(|v| !v.is_empty())?;
    Some(PathBuf::from(path))
}

/// The objective gradient anneal descends on, against a central difference
/// of the same objective's energy, and one steepest-descent step on it.
fn assert_objective_is_the_energy_slope(label: &str, pot: &RgpotObjective, x: &[f64]) {
    let objective = pot.wrapper();
    let (energy, gradient) = objective.value_and_gradient(ArrayView1::from(x));
    let gradient = gradient.to_vec();
    let fd = central_difference(&mut |y: &[f64]| objective.eval(ArrayView1::from(y)), x);
    let residual = max_residual(&gradient, &fd);
    let flipped = max_residual(&gradient, &fd.iter().map(|v| -v).collect::<Vec<_>>());
    eprintln!(
        "{label}: E {energy:.8} eV |fd| {:.6} eV/A max|g - fd| {residual:.3e} max|g + fd| {flipped:.3e}",
        norm(&fd)
    );
    assert!(
        residual < TOLERANCE,
        "{label}: objective gradient is not dE/dx (max|g - fd| {residual:.3e}, max|g + fd| {flipped:.3e} eV/A)"
    );

    let fresh = pot.fresh_evaluation(x).expect("fresh evaluation");
    let force_residual = max_residual(&fresh.forces, &fd.iter().map(|v| -v).collect::<Vec<_>>());
    assert!(
        force_residual < TOLERANCE,
        "{label}: reported forces are not -dE/dx (max|F + fd| {force_residual:.3e} eV/A)"
    );

    let step = 0.01 / norm(&gradient);
    let downhill: Array1<f64> = x
        .iter()
        .zip(&gradient)
        .map(|(xi, gi)| xi - step * gi)
        .collect();
    let lowered = objective.eval(downhill.view());
    assert!(
        lowered < energy,
        "{label}: a 0.01 A step against the objective gradient raised E from {energy} to {lowered}"
    );
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

#[repr(C)]
struct XtbConfig {
    method: i32,
    accuracy: f64,
    electronic_temperature: f64,
    max_iterations: i32,
    charge: f64,
    uhf: i32,
}

/// `rgpot_xtb_force` writes the force, -dE/dx in eV/Å, into its buffer.
#[test]
fn the_rgpot_xtb_engine_writes_minus_the_energy_slope() {
    let Some(path) = engine_path() else {
        eprintln!("RGPOT_XTB_ENGINE is unset; the engine sign is not checked");
        return;
    };
    let (z, x) = displaced_dimer();
    let n = z.len();
    let lib = unsafe { Library::new(&path) }.expect("dlopen RGPOT_XTB_ENGINE");
    let create: Symbol<XtbCreate> = unsafe { lib.get(b"rgpot_xtb_create\0") }.unwrap();
    let destroy: Symbol<XtbDestroy> = unsafe { lib.get(b"rgpot_xtb_destroy\0") }.unwrap();
    let force: Symbol<XtbForce> = unsafe { lib.get(b"rgpot_xtb_force\0") }.unwrap();
    let cfg = XtbConfig {
        method: 3,
        accuracy: anneal_core::catalog::molecular::GFN2_ACCURACY,
        electronic_temperature: 300.0,
        max_iterations: anneal_core::catalog::molecular::GFN2_MAX_ITERATIONS,
        charge: 0.0,
        uhf: 0,
    };
    let mut err = [0u8; 256];
    let pot = unsafe { create(&cfg, err.as_mut_ptr(), err.len()) };
    assert!(!pot.is_null(), "{}", String::from_utf8_lossy(&err));
    let call = |y: &[f64], buffer: &mut [f64]| {
        let mut energy = 0.0;
        let mut variance = 0.0;
        let rc = unsafe {
            force(
                pot,
                n as i64,
                y.as_ptr(),
                z.as_ptr(),
                buffer.as_mut_ptr(),
                &mut energy,
                &mut variance,
                BOX.as_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_xtb_force");
        energy
    };
    let mut buffer = vec![0.0; 3 * n];
    call(&x, &mut buffer);
    let fd = central_difference(&mut |y: &[f64]| call(y, &mut vec![0.0; 3 * n]), &x);
    unsafe { destroy(pot) };
    let minus_fd: Vec<f64> = fd.iter().map(|v| -v).collect();
    let residual = max_residual(&buffer, &minus_fd);
    eprintln!(
        "engine buffer: max|buffer + fd| {residual:.3e} max|buffer - fd| {:.3e} eV/A",
        max_residual(&buffer, &fd)
    );
    assert!(
        residual < TOLERANCE,
        "rgpot_xtb_force buffer is not -dE/dx (max|buffer + fd| {residual:.3e} eV/A)"
    );
}

#[test]
fn gfn2_objective_gradient_is_the_energy_slope_on_a_displaced_water_dimer() {
    let Some(path) = engine_path() else {
        eprintln!("RGPOT_XTB_ENGINE is unset; the GFN2 gradient is not checked");
        return;
    };
    assert!(
        path.exists(),
        "RGPOT_XTB_ENGINE {} is missing",
        path.display()
    );
    let (z, x) = displaced_dimer();
    assert_objective_is_the_energy_slope("water dimer", &RgpotObjective::xtb(&z, BOX), &x);
    assert_objective_is_the_energy_slope(
        "water dimer, fresh SCC",
        &RgpotObjective::xtb_fresh(&z, BOX),
        &x,
    );
}

#[test]
fn gfn2_objective_gradient_is_the_energy_slope_on_a_water_trimer() {
    let Some(path) = engine_path() else {
        eprintln!("RGPOT_XTB_ENGINE is unset; the GFN2 gradient is not checked");
        return;
    };
    assert!(
        path.exists(),
        "RGPOT_XTB_ENGINE {} is missing",
        path.display()
    );
    let (z, x) = trimer();
    assert_objective_is_the_energy_slope("water trimer", &RgpotObjective::xtb(&z, BOX), &x);
}

/// A harmonic pair energy behind the rgpot xtb engine ABI. Like
/// `rgpot_xtb_force`, it writes the force.
const STUB_ENGINE: &str = r#"
#include <math.h>
#include <stdlib.h>
void *rgpot_xtb_create(const void *cfg, char *err, unsigned long len) {
  (void)cfg; (void)err; (void)len; return malloc(1);
}
void rgpot_xtb_destroy(void *p) { free(p); }
void rgpot_xtb_reset(void *p) { (void)p; }
static double warm_move;
void rgpot_xtb_set_warm_move(void *p, double a) { (void)p; warm_move = a; }
double rgpot_xtb_warm_move(const void *p) { (void)p; return warm_move; }
int rgpot_xtb_force(void *p, long n, const double *x, const int *z,
                    double *f, double *e, double *var, const double *box) {
  (void)p; (void)z; (void)box;
  *e = 0.0; *var = 0.0;
  for (long i = 0; i < 3 * n; ++i) f[i] = 0.0;
  for (long i = 0; i < n; ++i)
    for (long j = i + 1; j < n; ++j) {
      double d[3], r = 0.0;
      for (int k = 0; k < 3; ++k) { d[k] = x[3*i+k] - x[3*j+k]; r += d[k]*d[k]; }
      r = sqrt(r);
      double s = r - 1.5;
      *e += 0.5 * 3.0 * s * s;
      for (int k = 0; k < 3; ++k) {
        double g = 3.0 * s * d[k] / r;
        f[3*i+k] -= g;
        f[3*j+k] += g;
      }
    }
  return 0;
}
"#;

fn build_stub_engine(dir: &Path) -> PathBuf {
    let source = dir.join("stub_engine.c");
    let library = dir.join("libstub_engine.so");
    std::fs::write(&source, STUB_ENGINE).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(cc)
        .args(["-O1", "-shared", "-fPIC", "-o"])
        .arg(&library)
        .arg(&source)
        .arg("-lm")
        .status()
        .expect("a C compiler for the stub engine");
    assert!(status.success(), "stub engine did not compile");
    library
}

/// The objective tests above, in a child whose engine is the stub.
#[test]
fn the_callback_passes_an_engine_force_through_to_an_energy_slope_gradient() {
    let dir = tempfile::tempdir().unwrap();
    let library = build_stub_engine(dir.path());
    for name in [
        "gfn2_objective_gradient_is_the_energy_slope_on_a_displaced_water_dimer",
        "gfn2_objective_gradient_is_the_energy_slope_on_a_water_trimer",
    ] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact", "--nocapture", "--test-threads=1"])
            .env("RGPOT_XTB_ENGINE", &library)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{name} on the stub engine:\n{log}");
        assert!(log.contains("1 passed"), "{name} did not run:\n{log}");
        assert!(
            !log.contains("is unset"),
            "{name} did not load the stub:\n{log}"
        );
    }
}
