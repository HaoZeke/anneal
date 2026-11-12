//! Build an eindir objective from an rgpot potential.
//!
//! Anneal does not evaluate XTB or CuH2. rgpot owns the kernel; eindir
//! owns the objective surface; this file only constructs the IS-A handle
//! the search already knows how to take.

#![allow(dead_code)]

use anneal_core::catalog::FreshEvaluation;
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
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Pair distance below which a geometry is an overlap, not a physical contact.
/// Same threshold the cluster walk uses before it will record a molecular quench.
const OVERLAP_SEPARATION_A: f64 = 0.35;
/// Same gate as `OVERLAP_CALL_LIMIT`. Below a covalent O–H, above the shell
/// where xtb's SCC fails.
const OVERLAP_CALL_LIMIT_A: f64 = 0.90;

static SCC_REJECTED: AtomicUsize = AtomicUsize::new(0);
static GEOMETRY_REJECTED: AtomicUsize = AtomicUsize::new(0);
static INSIDE_NS: AtomicU64 = AtomicU64::new(0);
static POTENTIAL_CALLS: AtomicUsize = AtomicUsize::new(0);
static WARM_CALLS: AtomicUsize = AtomicUsize::new(0);

/// How many xtb calls were rejected because the SCC did not return a finite result.
pub(crate) fn scc_rejected_count() -> usize {
    SCC_REJECTED.load(Ordering::Relaxed)
}

/// How many calls were refused before xtb because two atoms occupied the same place.
pub(crate) fn geometry_rejected_count() -> usize {
    GEOMETRY_REJECTED.load(Ordering::Relaxed)
}

/// Time spent inside `xtb_singlepoint`, summed over this process.
pub(crate) fn potential_inside_ns() -> u64 {
    INSIDE_NS.load(Ordering::Relaxed)
}

/// Single points that reached xtb, including those whose SCC was rejected.
pub(crate) fn potential_calls() -> usize {
    POTENTIAL_CALLS.load(Ordering::Relaxed)
}

/// Single points that reused the previous SCC wavefunction.
pub(crate) fn warm_calls() -> usize {
    WARM_CALLS.load(Ordering::Relaxed)
}

fn note_scc_rejected(min_pair_a: f64) {
    let n = SCC_REJECTED.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!("scc_rejected {n} min_pair_A {min_pair_a:.6}");
    let _ = io::stderr().flush();
}

fn note_geometry_rejected(min_pair_a: f64) {
    let n = GEOMETRY_REJECTED.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!("geometry_rejected {n} min_pair_A {min_pair_a:.6}");
    let _ = io::stderr().flush();
}

fn min_pair_distance_a(positions: &[f64]) -> f64 {
    let n = positions.len() / 3;
    if n < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[3 * i] - positions[3 * j];
            let dy = positions[3 * i + 1] - positions[3 * j + 1];
            let dz = positions[3 * i + 2] - positions[3 * j + 2];
            best = best.min((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    best
}

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

type Cuh2Force =
    unsafe extern "C" fn(i32, *const f64, *const i32, *const f64, *mut f64, *mut f64) -> i32;

unsafe extern "C" {
    fn fork() -> i32;
    fn pipe(fds: *mut i32) -> i32;
    fn close(fd: i32) -> i32;
    fn read(fd: i32, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn kill(pid: i32, sig: i32) -> i32;
    fn _exit(code: i32) -> !;
}

fn write_full(fd: i32, mut bytes: &[u8]) -> bool {
    while !bytes.is_empty() {
        let n = unsafe { write(fd, bytes.as_ptr() as *const c_void, bytes.len()) };
        if n <= 0 {
            return false;
        }
        bytes = &bytes[n as usize..];
    }
    true
}

fn read_full(fd: i32, bytes: &mut [u8]) -> bool {
    let mut filled = 0;
    while filled < bytes.len() {
        let n = unsafe {
            read(
                fd,
                bytes[filled..].as_mut_ptr() as *mut c_void,
                bytes.len() - filled,
            )
        };
        if n <= 0 {
            return false;
        }
        filled += n as usize;
    }
    true
}

/// Child side of one xtb evaluator. `ERROR STOP` ends this process only.
///
/// The calculator is created here, after the fork, and kept for every
/// geometry this child sees.
unsafe fn xtb_worker_main(incoming: i32, outgoing: i32) -> ! {
    let mut session = match anneal_core::xtb_session::XtbSession::open() {
        Ok(session) => session,
        Err(err) => {
            eprintln!("xtb session: {err}");
            unsafe { _exit(1) };
        }
    };
    loop {
        let mut n_bytes = [0u8; 8];
        if !read_full(incoming, &mut n_bytes) {
            unsafe { _exit(0) };
        }
        let n = u64::from_ne_bytes(n_bytes);
        if n == 0 || n > 10_000 {
            unsafe { _exit(0) };
        }
        let n_atoms = n as usize;
        let mut positions = vec![0.0f64; 3 * n_atoms];
        let mut atomic_numbers = vec![0i32; n_atoms];
        let mut cell = [0.0f64; 9];
        let pos_bytes = unsafe {
            std::slice::from_raw_parts_mut(positions.as_mut_ptr() as *mut u8, positions.len() * 8)
        };
        let z_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                atomic_numbers.as_mut_ptr() as *mut u8,
                atomic_numbers.len() * 4,
            )
        };
        let cell_bytes =
            unsafe { std::slice::from_raw_parts_mut(cell.as_mut_ptr() as *mut u8, 9 * 8) };
        if !read_full(incoming, pos_bytes)
            || !read_full(incoming, z_bytes)
            || !read_full(incoming, cell_bytes)
        {
            unsafe { _exit(0) };
        }
        let mut gradient = vec![0.0f64; 3 * n_atoms];
        let answered = session.evaluate(&positions, &atomic_numbers, &cell, &mut gradient);
        let (rc, energy, singlepoint_ns, warm): (i32, f64, u64, u64) = match answered {
            Ok(point) => (
                0,
                point.energy_ev,
                point.singlepoint_ns,
                u64::from(point.reused_wavefunction),
            ),
            Err(err) => (-2, f64::NAN, err.singlepoint_ns, 0),
        };
        let mut message = Vec::with_capacity(4 + 8 + 8 + 8 + gradient.len() * 8);
        message.extend_from_slice(&rc.to_ne_bytes());
        message.extend_from_slice(&energy.to_ne_bytes());
        message.extend_from_slice(&singlepoint_ns.to_ne_bytes());
        message.extend_from_slice(&warm.to_ne_bytes());
        let grad_bytes = unsafe {
            std::slice::from_raw_parts(gradient.as_ptr() as *const u8, gradient.len() * 8)
        };
        message.extend_from_slice(grad_bytes);
        if !write_full(outgoing, &message) {
            unsafe { _exit(0) };
        }
    }
}

enum WorkerFail {
    Dead,
    Rejected,
}

struct XtbWorker {
    pid: i32,
    to_child: i32,
    from_child: i32,
}

impl XtbWorker {
    fn spawn() -> Self {
        unsafe {
            let mut to_child = [0i32; 2];
            let mut from_child = [0i32; 2];
            if pipe(to_child.as_mut_ptr()) != 0 || pipe(from_child.as_mut_ptr()) != 0 {
                panic!("xtb worker pipe failed");
            }
            let pid = fork();
            if pid < 0 {
                panic!("xtb worker fork failed");
            }
            if pid == 0 {
                close(to_child[1]);
                close(from_child[0]);
                xtb_worker_main(to_child[0], from_child[1]);
            }
            close(to_child[0]);
            close(from_child[1]);
            Self {
                pid,
                to_child: to_child[1],
                from_child: from_child[0],
            }
        }
    }

    fn shutdown(&mut self) {
        if self.pid <= 0 {
            return;
        }
        let pid = self.pid;
        self.pid = 0;
        unsafe {
            close(self.to_child);
            close(self.from_child);
            let mut status = 0;
            // WNOHANG: a child that already died from ERROR STOP must be reaped
            // without signalling a pid that the kernel may have reused.
            if waitpid(pid, &mut status, 1) == 0 {
                kill(pid, 9);
                waitpid(pid, &mut status, 0);
            }
        }
    }

    /// One energy and gradient. A dead child is replaced; the geometry is not retried.
    fn evaluate(
        &mut self,
        n: usize,
        positions: &[f64],
        atomic_numbers: &[i32],
        cell: &[f64],
        gradient: &mut [f64],
    ) -> Result<(f64, f64), WorkerFail> {
        match self.exchange(n, positions, atomic_numbers, cell, gradient) {
            Ok(answer) => Ok(answer),
            Err(WorkerFail::Rejected) => Err(WorkerFail::Rejected),
            Err(WorkerFail::Dead) => {
                self.shutdown();
                *self = Self::spawn();
                Err(WorkerFail::Dead)
            }
        }
    }

    fn exchange(
        &mut self,
        n: usize,
        positions: &[f64],
        atomic_numbers: &[i32],
        cell: &[f64],
        gradient: &mut [f64],
    ) -> Result<(f64, f64), WorkerFail> {
        let mut request =
            Vec::with_capacity(8 + positions.len() * 8 + atomic_numbers.len() * 4 + 72);
        request.extend_from_slice(&(n as u64).to_ne_bytes());
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, positions.len() * 8)
        };
        let z_bytes = unsafe {
            std::slice::from_raw_parts(
                atomic_numbers.as_ptr() as *const u8,
                atomic_numbers.len() * 4,
            )
        };
        let cell_bytes = unsafe { std::slice::from_raw_parts(cell.as_ptr() as *const u8, 72) };
        request.extend_from_slice(pos_bytes);
        request.extend_from_slice(z_bytes);
        request.extend_from_slice(cell_bytes);
        if !write_full(self.to_child, &request) {
            return Err(WorkerFail::Dead);
        }
        let mut header = [0u8; 4 + 8 + 8 + 8];
        if !read_full(self.from_child, &mut header) {
            return Err(WorkerFail::Dead);
        }
        let rc = i32::from_ne_bytes(header[0..4].try_into().unwrap());
        let energy = f64::from_ne_bytes(header[4..12].try_into().unwrap());
        let singlepoint_ns = u64::from_ne_bytes(header[12..20].try_into().unwrap());
        let warm = u64::from_ne_bytes(header[20..28].try_into().unwrap());
        INSIDE_NS.fetch_add(singlepoint_ns, Ordering::Relaxed);
        POTENTIAL_CALLS.fetch_add(1, Ordering::Relaxed);
        if warm == 1 {
            WARM_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let variance = 0.0;
        let grad_bytes = unsafe {
            std::slice::from_raw_parts_mut(gradient.as_mut_ptr() as *mut u8, gradient.len() * 8)
        };
        if !read_full(self.from_child, grad_bytes) {
            return Err(WorkerFail::Dead);
        }
        if rc != 0 || !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(WorkerFail::Rejected);
        }
        let _ = variance;
        Ok((energy, variance))
    }
}

struct XtbKernel {
    worker: XtbWorker,
    evaluations: usize,
    /// Force buffer reused across calls. The owned DLPack tensor copies
    /// it before the callback returns.
    scratch: Vec<f64>,
}

impl Drop for XtbKernel {
    fn drop(&mut self) {
        self.worker.shutdown();
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
    if kernel.scratch.len() != 3 * n {
        kernel.scratch.resize(3 * n, 0.0);
    }
    kernel.scratch.fill(0.0);
    kernel.evaluations += 1;
    let positions = unsafe { std::slice::from_raw_parts(pos, 3 * n) };
    let atomic_numbers = unsafe { std::slice::from_raw_parts(z, n) };
    let cell = unsafe { std::slice::from_raw_parts(boxp, 9) };
    let min_pair = min_pair_distance_a(positions);
    if min_pair <= OVERLAP_CALL_LIMIT_A {
        // xtb's SCC aborts when a trial step stacks two atoms. That geometry
        // is not an energy the search can record. The caller turns the error
        // into an uphill step and shortens it, so a finite quench stays eligible.
        note_geometry_rejected(min_pair);
        return rgpot_status_t::RGPOT_INTERNAL_ERROR;
    }
    if let Some(path) = env_path("RGPOT_XTB_TRACE")
        && let Ok(mut trace) = File::create(path)
    {
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
    let (energy, var) =
        match kernel
            .worker
            .evaluate(n, positions, atomic_numbers, cell, &mut kernel.scratch)
        {
            Ok(answer) => answer,
            Err(_) => {
                note_scc_rejected(min_pair);
                kernel.scratch.fill(f64::NAN);
                return rgpot_status_t::RGPOT_INTERNAL_ERROR;
            }
        };
    out.energy = energy;
    out.variance = var;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(kernel.scratch.as_ptr(), n as i64, 3) };
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
        let worker = XtbWorker::spawn();
        let kernel = Box::new(XtbKernel {
            worker,
            evaluations: 0,
            scratch: Vec::new(),
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
        println!("  eindir objective from libxtb");
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
