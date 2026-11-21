// MIT License
// Copyright 2023--present rgpot developers

//! Dense Hessians, dual-number callbacks, resolved schemes and the
//! derivative capabilities and counts, through the C ABI, against a Morse
//! cluster whose Hessian is known in closed form.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use dlpk::sys::DLManagedTensorVersioned;

use rgpot_core::c_api::hessian::{
    rgpot_potential_derivative_caps, rgpot_potential_derivative_counts, rgpot_potential_hessian,
    rgpot_potential_set_ad_hessian_callback, rgpot_potential_set_hessian_callback,
};
use rgpot_core::c_api::hvp::{
    rgpot_potential_hvp, rgpot_potential_hvp_counts, rgpot_potential_hvp_resolved,
    rgpot_potential_reset_hvp_counts, rgpot_potential_set_ad_hvp_callback,
    rgpot_potential_set_hvp_callback,
};
use rgpot_core::c_api::potential::{rgpot_potential_free, rgpot_potential_new};
use rgpot_core::c_api::types::rgpot_force_input_create;
use rgpot_core::hessian::{
    rgpot_derivative_caps_t, rgpot_derivative_counts_t, rgpot_hessian_scheme_t,
};
use rgpot_core::hvp::{rgpot_hvp_counts_t, rgpot_hvp_scheme_t};
use rgpot_core::potential::{rgpot_potential_t, PotentialCallback};
use rgpot_core::status::{rgpot_last_error, rgpot_set_last_error, rgpot_status_t};
use rgpot_core::tensor::rgpot_tensor_owned_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

use rgpot_hessian_scheme_t::*;
use rgpot_hvp_scheme_t::*;

struct Morse {
    de: f64,
    a: f64,
    re: f64,
    force_calls: AtomicUsize,
    analytic_calls: AtomicUsize,
    ad_calls: AtomicUsize,
}

impl Morse {
    fn platinum() -> Self {
        Self {
            de: 0.7102,
            a: 1.6047,
            re: 2.8970,
            force_calls: AtomicUsize::new(0),
            analytic_calls: AtomicUsize::new(0),
            ad_calls: AtomicUsize::new(0),
        }
    }

    /// The pair energy and its first two radial derivatives.
    fn pair(&self, r: f64) -> (f64, f64, f64) {
        let e = (-self.a * (r - self.re)).exp();
        (
            self.de * (1.0 - e) * (1.0 - e) - self.de,
            2.0 * self.de * self.a * e * (1.0 - e),
            2.0 * self.de * self.a * self.a * e * (2.0 * e - 1.0),
        )
    }

    fn for_pairs(&self, pos: &[f64], mut f: impl FnMut(usize, usize, [f64; 3], f64)) {
        let n = pos.len() / 3;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    pos[3 * i] - pos[3 * j],
                    pos[3 * i + 1] - pos[3 * j + 1],
                    pos[3 * i + 2] - pos[3 * j + 2],
                ];
                f(i, j, d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
    }

    fn energy_forces(&self, pos: &[f64]) -> (f64, Vec<f64>) {
        let mut energy = 0.0;
        let mut forces = vec![0.0; pos.len()];
        self.for_pairs(pos, |i, j, d, r| {
            let (phi, dphi, _) = self.pair(r);
            energy += phi;
            for k in 0..3 {
                forces[3 * i + k] -= dphi * d[k] / r;
                forces[3 * j + k] += dphi * d[k] / r;
            }
        });
        (energy, forces)
    }

    /// The dense Hessian from the pair blocks
    /// `K = phi'' u u^T + (phi' / r) (I - u u^T)`.
    fn hessian(&self, pos: &[f64]) -> Vec<f64> {
        let dim = pos.len();
        let mut h = vec![0.0; dim * dim];
        self.for_pairs(pos, |i, j, d, r| {
            let (_, dphi, d2phi) = self.pair(r);
            for p in 0..3 {
                for q in 0..3 {
                    let uu = d[p] * d[q] / (r * r);
                    let eye = if p == q { 1.0 } else { 0.0 };
                    let k = d2phi * uu + dphi / r * (eye - uu);
                    h[(3 * i + p) * dim + 3 * i + q] += k;
                    h[(3 * j + p) * dim + 3 * j + q] += k;
                    h[(3 * i + p) * dim + 3 * j + q] -= k;
                    h[(3 * j + p) * dim + 3 * i + q] -= k;
                }
            }
        });
        h
    }
}

fn positions(input: &rgpot_force_input_t) -> Vec<f64> {
    let n = unsafe { input.n_atoms() }.unwrap();
    unsafe { std::slice::from_raw_parts((*input.positions).dl_tensor.data as *const f64, 3 * n) }
        .to_vec()
}

fn multiply(h: &[f64], v: &[f64]) -> Vec<f64> {
    let dim = v.len();
    (0..dim)
        .map(|r| (0..dim).map(|c| h[r * dim + c] * v[c]).sum())
        .collect()
}

unsafe extern "C" fn morse_force(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.force_calls.fetch_add(1, Ordering::SeqCst);
    let pos = positions(unsafe { &*input });
    let (energy, forces) = morse.energy_forces(&pos);
    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces =
        unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), (pos.len() / 3) as i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

static OWNED_DELETER: OnceLock<unsafe extern "C" fn(*mut DLManagedTensorVersioned)> =
    OnceLock::new();

unsafe extern "C" fn messaging_deleter(tensor: *mut DLManagedTensorVersioned) {
    unsafe { rgpot_set_last_error(c"a deleter's message".as_ptr()) };
    let owned = OWNED_DELETER.get().expect("stored before the swap");
    unsafe { owned(tensor) };
}

/// [`morse_force`], whose force tensor stores a message when it is freed.
unsafe extern "C" fn messaging_force(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let status = unsafe { morse_force(user_data, input, output) };
    let forces = unsafe { &mut *(*output).forces };
    OWNED_DELETER.get_or_init(|| forces.deleter.expect("an owned tensor has a deleter"));
    forces.deleter = Some(messaging_deleter);
    status
}

unsafe fn write_hessian(
    morse: &Morse,
    input: *const rgpot_force_input_t,
    out: *mut f64,
) -> rgpot_status_t {
    let h = morse.hessian(&positions(unsafe { &*input }));
    unsafe { std::slice::from_raw_parts_mut(out, h.len()) }.copy_from_slice(&h);
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe fn write_product(
    morse: &Morse,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    out: *mut f64,
) -> rgpot_status_t {
    let pos = positions(unsafe { &*input });
    let v = unsafe { std::slice::from_raw_parts(direction, pos.len()) };
    let hv = multiply(&morse.hessian(&pos), v);
    unsafe { std::slice::from_raw_parts_mut(out, hv.len()) }.copy_from_slice(&hv);
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn analytic_hessian(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    out: *mut f64,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.analytic_calls.fetch_add(1, Ordering::SeqCst);
    unsafe { write_hessian(morse, input, out) }
}

unsafe extern "C" fn ad_hessian(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    out: *mut f64,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.ad_calls.fetch_add(1, Ordering::SeqCst);
    unsafe { write_hessian(morse, input, out) }
}

unsafe extern "C" fn analytic_hvp(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    out: *mut f64,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.analytic_calls.fetch_add(1, Ordering::SeqCst);
    unsafe { write_product(morse, input, direction, out) }
}

unsafe extern "C" fn ad_hvp(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    out: *mut f64,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.ad_calls.fetch_add(1, Ordering::SeqCst);
    unsafe { write_product(morse, input, direction, out) }
}

unsafe extern "C" fn failing_hessian(
    _: *mut c_void,
    _: *const rgpot_force_input_t,
    _: *mut f64,
) -> rgpot_status_t {
    unsafe { rgpot_set_last_error(c"the SCC did not converge".as_ptr()) };
    rgpot_status_t::RGPOT_SCC_NOT_CONVERGED
}

unsafe extern "C" fn silent_hessian(
    _: *mut c_void,
    _: *const rgpot_force_input_t,
    _: *mut f64,
) -> rgpot_status_t {
    rgpot_status_t::RGPOT_INTERNAL_ERROR
}

unsafe extern "C" fn silent_hvp(
    _: *mut c_void,
    _: *const rgpot_force_input_t,
    _: *const f64,
    _: *mut f64,
) -> rgpot_status_t {
    rgpot_status_t::RGPOT_INTERNAL_ERROR
}

/// Forces of +-1.5e308 on a coordinate moved up or down from the positions
/// behind `user_data`, so that every central difference overflows.
unsafe extern "C" fn cliff_force(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let x0 = unsafe { &*(user_data as *const Vec<f64>) };
    let pos = positions(unsafe { &*input });
    let forces: Vec<f64> = pos
        .iter()
        .zip(x0)
        .map(|(x, x0)| {
            if x > x0 {
                1.5e308
            } else if x < x0 {
                -1.5e308
            } else {
                0.0
            }
        })
        .collect();
    let out = unsafe { &mut *output };
    out.energy = 0.0;
    out.variance = 0.0;
    out.forces =
        unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), (pos.len() / 3) as i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn nan_hessian(
    _: *mut c_void,
    input: *const rgpot_force_input_t,
    out: *mut f64,
) -> rgpot_status_t {
    let dim = 3 * unsafe { (*input).n_atoms() }.unwrap();
    let h = unsafe { std::slice::from_raw_parts_mut(out, dim * dim) };
    h.fill(0.0);
    h[dim + 2] = f64::NAN;
    rgpot_status_t::RGPOT_SUCCESS
}

/// Six atoms on a distorted octahedron of edge about 2.83 A, inside a
/// 30 A cell the model ignores.
struct System {
    pos: Vec<f64>,
    z: Vec<i32>,
    cell: [f64; 9],
}

impl System {
    fn octahedron() -> Self {
        let offsets = [
            2.00, 0.10, -0.05, //
            -1.95, -0.08, 0.12, //
            0.07, 2.03, 0.02, //
            -0.10, -2.05, -0.06, //
            0.04, -0.03, 1.98, //
            0.02, 0.09, -2.02,
        ];
        Self {
            pos: offsets.iter().map(|x| 15.0 + x).collect(),
            z: vec![78; 6],
            cell: [30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0],
        }
    }

    fn input(&mut self) -> rgpot_force_input_t {
        unsafe {
            rgpot_force_input_create(
                self.z.len(),
                self.pos.as_mut_ptr(),
                self.z.as_mut_ptr(),
                self.cell.as_mut_ptr(),
            )
        }
    }

    fn dim(&self) -> usize {
        self.pos.len()
    }
}

struct Handle {
    pot: *mut rgpot_potential_t,
    morse: Box<Morse>,
}

impl Handle {
    fn new() -> Self {
        Self::with_force(morse_force)
    }

    fn with_force(force: PotentialCallback) -> Self {
        let morse = Box::new(Morse::platinum());
        let pot =
            unsafe { rgpot_potential_new(force, &*morse as *const Morse as *mut c_void, None) };
        assert!(!pot.is_null());
        Self { pot, morse }
    }

    fn calls(counter: &AtomicUsize) -> usize {
        counter.load(Ordering::SeqCst)
    }

    fn force_calls(&self) -> usize {
        Self::calls(&self.morse.force_calls)
    }

    fn caps(&self) -> rgpot_derivative_caps_t {
        let mut out = std::mem::MaybeUninit::<rgpot_derivative_caps_t>::uninit();
        let status = unsafe { rgpot_potential_derivative_caps(self.pot, out.as_mut_ptr()) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        unsafe { out.assume_init() }
    }

    fn counts(&self) -> rgpot_derivative_counts_t {
        let mut out = rgpot_derivative_counts_t::default();
        let status = unsafe { rgpot_potential_derivative_counts(self.pot, &mut out) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        out
    }

    fn hvp_counts(&self) -> rgpot_hvp_counts_t {
        let mut out = rgpot_hvp_counts_t::default();
        let status = unsafe { rgpot_potential_hvp_counts(self.pot, &mut out) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        out
    }

    fn try_hvp(
        &self,
        input: &rgpot_force_input_t,
        v: &[f64],
        scheme: i32,
    ) -> Result<(Vec<f64>, rgpot_hvp_scheme_t), rgpot_status_t> {
        let mut hv = vec![f64::NAN; v.len()];
        let mut used = RGPOT_HVP_AUTO;
        let status = unsafe {
            rgpot_potential_hvp_resolved(
                self.pot,
                input,
                v.as_ptr(),
                0.0,
                scheme,
                hv.as_mut_ptr(),
                &mut used,
            )
        };
        if status == rgpot_status_t::RGPOT_SUCCESS {
            Ok((hv, used))
        } else {
            assert_eq!(used, RGPOT_HVP_AUTO, "scheme_used written on failure");
            Err(status)
        }
    }

    fn try_hessian(
        &self,
        input: &rgpot_force_input_t,
        step: f64,
        scheme: i32,
    ) -> Result<(Vec<f64>, rgpot_hessian_scheme_t), rgpot_status_t> {
        let dim = 3 * unsafe { input.n_atoms() }.unwrap();
        let mut h = vec![f64::NAN; dim * dim];
        let mut used = RGPOT_HESSIAN_CENTRAL;
        let status = unsafe {
            rgpot_potential_hessian(self.pot, input, step, scheme, h.as_mut_ptr(), &mut used)
        };
        if status == rgpot_status_t::RGPOT_SUCCESS {
            Ok((h, used))
        } else {
            assert_eq!(
                used, RGPOT_HESSIAN_CENTRAL,
                "scheme_used written on failure"
            );
            Err(status)
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free(self.pot) };
    }
}

fn last_error() -> String {
    unsafe { std::ffi::CStr::from_ptr(rgpot_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn direction(dim: usize) -> Vec<f64> {
    (0..dim).map(|k| (0.9 * k as f64 + 0.2).cos()).collect()
}

fn largest(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn exactly_symmetric(h: &[f64], dim: usize) -> bool {
    (0..dim).all(|r| ((r + 1)..dim).all(|c| h[r * dim + c] == h[c * dim + r]))
}

#[test]
fn caps_follow_the_registered_callbacks() {
    let handle = Handle::new();
    let caps = handle.caps();
    assert!(!caps.hvp_analytic && !caps.hvp_ad && caps.hvp_fd);
    assert!(!caps.hessian_analytic && !caps.hessian_ad && caps.hessian_fd);
    assert_eq!(caps.hvp_forward_force_calls, 1);
    assert_eq!(caps.hvp_central_force_calls, 2);
    assert_eq!(caps.hessian_central_force_calls_per_atom, 6);

    let pot = handle.pot;
    unsafe {
        assert_eq!(
            rgpot_potential_set_hvp_callback(pot, Some(analytic_hvp)),
            rgpot_status_t::RGPOT_SUCCESS
        );
        assert_eq!(
            rgpot_potential_set_ad_hvp_callback(pot, Some(ad_hvp)),
            rgpot_status_t::RGPOT_SUCCESS
        );
        assert_eq!(
            rgpot_potential_set_hessian_callback(pot, Some(analytic_hessian)),
            rgpot_status_t::RGPOT_SUCCESS
        );
        assert_eq!(
            rgpot_potential_set_ad_hessian_callback(pot, Some(ad_hessian)),
            rgpot_status_t::RGPOT_SUCCESS
        );
    }
    let caps = handle.caps();
    assert!(caps.hvp_analytic && caps.hvp_ad && caps.hessian_analytic && caps.hessian_ad);

    unsafe {
        rgpot_potential_set_hvp_callback(pot, None);
        rgpot_potential_set_ad_hessian_callback(pot, None);
    }
    let caps = handle.caps();
    assert!(!caps.hvp_analytic && caps.hvp_ad && caps.hessian_analytic && !caps.hessian_ad);

    let null = std::ptr::null();
    let mut out = caps;
    unsafe {
        assert_eq!(
            rgpot_potential_derivative_caps(null, &mut out),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_derivative_caps(pot, std::ptr::null_mut()),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_derivative_counts(pot, std::ptr::null_mut()),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_set_hessian_callback(null, None),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_set_ad_hessian_callback(null, None),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_set_ad_hvp_callback(null, None),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
    }
}

#[test]
fn auto_reports_its_fallback_to_forward() {
    let mut system = System::octahedron();
    let input = system.input();
    let v = direction(system.dim());
    let handle = Handle::new();

    let (_, used) = handle.try_hvp(&input, &v, RGPOT_HVP_AUTO as i32).unwrap();
    assert_eq!(used, RGPOT_HVP_FORWARD);
    let (_, used) = handle
        .try_hvp(&input, &v, RGPOT_HVP_CENTRAL as i32)
        .unwrap();
    assert_eq!(used, RGPOT_HVP_CENTRAL);
    let counts = handle.counts();
    assert_eq!((counts.hvp_forward, counts.hvp_central), (1, 1));
    assert_eq!((counts.hvp_analytic, counts.hvp_ad), (0, 0));

    unsafe { rgpot_potential_set_hvp_callback(handle.pot, Some(analytic_hvp)) };
    let (_, used) = handle.try_hvp(&input, &v, RGPOT_HVP_AUTO as i32).unwrap();
    assert_eq!(used, RGPOT_HVP_ANALYTIC);
    assert_eq!(handle.counts().hvp_analytic, 1);

    // The product without the out-parameter counts the scheme it ran too.
    let mut hv = vec![0.0; v.len()];
    let status = unsafe {
        rgpot_potential_hvp(
            handle.pot,
            &input,
            v.as_ptr(),
            0.0,
            RGPOT_HVP_EXACT as i32,
            hv.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(handle.counts().hvp_analytic, 2);
    assert_eq!(handle.hvp_counts().products, 4);
}

#[test]
fn exact_and_ad_refuse_without_a_call() {
    let mut system = System::octahedron();
    let input = system.input();
    let v = direction(system.dim());
    let handle = Handle::new();

    for (scheme, name) in [
        (RGPOT_HVP_EXACT as i32, "RGPOT_HVP_EXACT"),
        (RGPOT_HVP_AD as i32, "RGPOT_HVP_AD"),
        (RGPOT_HVP_ANALYTIC as i32, "RGPOT_HVP_ANALYTIC"),
    ] {
        assert_eq!(
            handle.try_hvp(&input, &v, scheme).unwrap_err(),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert!(last_error().starts_with(name), "{}", last_error());
    }
    for (scheme, name) in [
        (RGPOT_HESSIAN_EXACT as i32, "RGPOT_HESSIAN_EXACT"),
        (RGPOT_HESSIAN_AD as i32, "RGPOT_HESSIAN_AD"),
        (RGPOT_HESSIAN_ANALYTIC as i32, "RGPOT_HESSIAN_ANALYTIC"),
    ] {
        assert_eq!(
            handle.try_hessian(&input, 0.0, scheme).unwrap_err(),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert!(last_error().starts_with(name), "{}", last_error());
    }

    unsafe { rgpot_potential_set_hvp_callback(handle.pot, Some(analytic_hvp)) };
    unsafe { rgpot_potential_set_hessian_callback(handle.pot, Some(analytic_hessian)) };
    assert!(handle.try_hvp(&input, &v, RGPOT_HVP_AD as i32).is_err());
    assert!(handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_AD as i32)
        .is_err());

    assert_eq!(handle.force_calls(), 0);
    assert_eq!(Handle::calls(&handle.morse.analytic_calls), 0);
    assert_eq!(handle.counts(), rgpot_derivative_counts_t::default());
    assert_eq!(handle.hvp_counts().products, 0);
}

#[test]
fn exact_takes_the_dual_number_callbacks_when_alone() {
    let mut system = System::octahedron();
    let input = system.input();
    let v = direction(system.dim());
    let handle = Handle::new();
    unsafe {
        rgpot_potential_set_ad_hvp_callback(handle.pot, Some(ad_hvp));
        rgpot_potential_set_ad_hessian_callback(handle.pot, Some(ad_hessian));
    }
    let want = handle.morse.hessian(&system.pos);

    let (hv, used) = handle.try_hvp(&input, &v, RGPOT_HVP_EXACT as i32).unwrap();
    assert_eq!(used, RGPOT_HVP_AD);
    assert_eq!(hv, multiply(&want, &v));
    let (h, used) = handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_EXACT as i32)
        .unwrap();
    assert_eq!(used, RGPOT_HESSIAN_AD);
    assert_eq!(h, want);
    let (_, used) = handle.try_hvp(&input, &v, RGPOT_HVP_AUTO as i32).unwrap();
    assert_eq!(used, RGPOT_HVP_FORWARD, "AUTO never takes the dual numbers");

    let counts = handle.counts();
    assert_eq!(
        (counts.hvp_ad, counts.hessian_ad, counts.hvp_forward),
        (1, 1, 1)
    );
    assert_eq!(Handle::calls(&handle.morse.ad_calls), 2);
    assert_eq!(handle.hvp_counts().analytic, 0);
}

#[test]
fn the_analytic_hessian_is_the_callback_and_makes_no_force_call() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    unsafe {
        rgpot_potential_set_hessian_callback(handle.pot, Some(analytic_hessian));
        rgpot_potential_set_ad_hessian_callback(handle.pot, Some(ad_hessian));
    }
    let (h, used) = handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_EXACT as i32)
        .unwrap();
    assert_eq!(used, RGPOT_HESSIAN_ANALYTIC);
    assert_eq!(h, handle.morse.hessian(&system.pos));
    let (_, used) = handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_AD as i32)
        .unwrap();
    assert_eq!(used, RGPOT_HESSIAN_AD);
    assert_eq!(handle.force_calls(), 0);
    let counts = handle.counts();
    assert_eq!((counts.hessian_analytic, counts.hessian_ad), (1, 1));
    assert_eq!(counts.hessian_force_calls, 0);
}

#[test]
fn a_central_hessian_costs_6n_calls_and_converges_at_second_order() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    let want = handle.morse.hessian(&system.pos);
    let scale = largest(&want);
    let dim = system.dim();

    let (h, used) = handle
        .try_hessian(&input, 1e-4, RGPOT_HESSIAN_CENTRAL as i32)
        .unwrap();
    assert_eq!(used, RGPOT_HESSIAN_CENTRAL);
    assert_eq!(handle.force_calls(), 6 * 6);
    let counts = handle.counts();
    assert_eq!(counts.hessian_central, 1);
    assert_eq!(counts.hessian_force_calls, 36);
    assert_eq!(handle.hvp_counts().force_calls, 0);
    assert!(exactly_symmetric(&h, dim));
    assert!(largest_difference(&h, &want) < 1e-7 * scale);

    let error = |step: f64| {
        let (h, _) = handle
            .try_hessian(&input, step, RGPOT_HESSIAN_CENTRAL as i32)
            .unwrap();
        largest_difference(&h, &want) / scale
    };
    let order = (error(2e-2) / error(1e-2)).log2();
    assert!((order - 2.0).abs() < 0.1, "order {order}");
}

#[test]
fn hessian_refuses_malformed_requests() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    let dim = system.dim();
    let mut h = vec![0.0; dim * dim];
    let mut used = RGPOT_HESSIAN_CENTRAL;
    let central = RGPOT_HESSIAN_CENTRAL as i32;
    unsafe {
        let null_pot = std::ptr::null();
        assert_eq!(
            rgpot_potential_hessian(null_pot, &input, 0.0, central, h.as_mut_ptr(), &mut used),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_hessian(
                handle.pot,
                std::ptr::null(),
                0.0,
                central,
                h.as_mut_ptr(),
                &mut used
            ),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_hessian(
                handle.pot,
                &input,
                0.0,
                central,
                std::ptr::null_mut(),
                &mut used
            ),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            rgpot_potential_hessian(
                handle.pot,
                &input,
                0.0,
                central,
                h.as_mut_ptr(),
                std::ptr::null_mut()
            ),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
    }
    for raw in [0, 1, 6, -1] {
        assert_eq!(
            handle.try_hessian(&input, 0.0, raw).unwrap_err(),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert!(last_error().contains("unknown scheme"), "{}", last_error());
    }
    for step in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            handle.try_hessian(&input, step, central).unwrap_err(),
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
    }
    let v = direction(dim);
    let mut hv = vec![0.0; dim];
    let status = unsafe {
        rgpot_potential_hvp_resolved(
            handle.pot,
            &input,
            v.as_ptr(),
            0.0,
            RGPOT_HVP_AUTO as i32,
            hv.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(
        handle.try_hvp(&input, &v, 6).unwrap_err(),
        rgpot_status_t::RGPOT_INVALID_PARAMETER
    );
    assert_eq!(handle.force_calls(), 0);
}

#[test]
fn callback_failures_keep_their_status_and_message() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    unsafe { rgpot_potential_set_hessian_callback(handle.pot, Some(failing_hessian)) };
    assert_eq!(
        handle
            .try_hessian(&input, 0.0, RGPOT_HESSIAN_ANALYTIC as i32)
            .unwrap_err(),
        rgpot_status_t::RGPOT_SCC_NOT_CONVERGED
    );
    assert_eq!(last_error(), "the SCC did not converge");

    unsafe { rgpot_potential_set_hessian_callback(handle.pot, Some(nan_hessian)) };
    assert_eq!(
        handle
            .try_hessian(&input, 0.0, RGPOT_HESSIAN_EXACT as i32)
            .unwrap_err(),
        rgpot_status_t::RGPOT_INVALID_PARAMETER
    );
    assert_eq!(
        last_error(),
        "Hessian callback: non-finite value at element 20"
    );
    assert_eq!(handle.counts().hessian_analytic, 0);
    assert_eq!(handle.force_calls(), 0);
}

#[test]
fn a_callback_that_sets_no_message_is_not_given_an_older_one() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    unsafe {
        rgpot_potential_set_hessian_callback(handle.pot, Some(analytic_hessian));
        rgpot_set_last_error(c"an older failure".as_ptr());
    }
    assert!(handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_ANALYTIC as i32)
        .is_ok());
    assert_eq!(last_error(), "an older failure");

    unsafe {
        rgpot_potential_set_hessian_callback(handle.pot, Some(silent_hessian));
        rgpot_potential_set_hvp_callback(handle.pot, Some(silent_hvp));
        rgpot_set_last_error(c"an older failure".as_ptr());
    }
    assert_eq!(
        handle
            .try_hessian(&input, 0.0, RGPOT_HESSIAN_ANALYTIC as i32)
            .unwrap_err(),
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert_eq!(
        last_error(),
        "Hessian callback returned RGPOT_INTERNAL_ERROR"
    );

    unsafe { rgpot_set_last_error(c"an older failure".as_ptr()) };
    assert_eq!(
        handle
            .try_hvp(&input, &direction(system.dim()), RGPOT_HVP_ANALYTIC as i32)
            .unwrap_err(),
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert_eq!(
        last_error(),
        "analytic HVP callback returned RGPOT_INTERNAL_ERROR"
    );
}

#[test]
fn the_rust_hessian_checks_the_output_length_before_its_input() {
    let mut system = System::octahedron();
    system.pos[0] = f64::NAN;
    let handle = Handle::new();
    let dim = system.dim();
    let mut short = vec![0.0; 9];
    let input = system.input();
    let err = unsafe { (*handle.pot).hessian(&input, 0.0, RGPOT_HESSIAN_CENTRAL, &mut short) }
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("hessian_out: expected {} elements, got 9", dim * dim)
    );

    let inside = unsafe { short.as_mut_ptr().add(1) };
    let empty =
        unsafe { rgpot_force_input_create(0, inside, std::ptr::null_mut(), std::ptr::null_mut()) };
    let err = unsafe { (*handle.pot).hessian(&empty, 0.0, RGPOT_HESSIAN_CENTRAL, &mut short) }
        .unwrap_err();
    assert_eq!(err.to_string(), "hessian_out: expected 0 elements, got 9");
    assert_eq!(handle.force_calls(), 0);
}

#[test]
fn a_successful_finite_difference_keeps_the_earlier_message() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    for scheme in [RGPOT_HVP_FORWARD, RGPOT_HVP_CENTRAL] {
        unsafe { rgpot_set_last_error(c"an older failure".as_ptr()) };
        assert!(handle
            .try_hvp(&input, &direction(system.dim()), scheme as i32)
            .is_ok());
        assert_eq!(last_error(), "an older failure");
    }
    unsafe { rgpot_set_last_error(c"an older failure".as_ptr()) };
    assert!(handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_CENTRAL as i32)
        .is_ok());
    assert_eq!(last_error(), "an older failure");
    assert!(handle.force_calls() > 0);
}

#[test]
fn a_force_tensor_deleter_does_not_replace_the_earlier_message() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::with_force(messaging_force);
    for scheme in [RGPOT_HVP_FORWARD, RGPOT_HVP_CENTRAL] {
        unsafe { rgpot_set_last_error(c"an older failure".as_ptr()) };
        assert!(handle
            .try_hvp(&input, &direction(system.dim()), scheme as i32)
            .is_ok());
        assert_eq!(last_error(), "an older failure");
    }
    unsafe { rgpot_set_last_error(c"an older failure".as_ptr()) };
    assert!(handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_CENTRAL as i32)
        .is_ok());
    assert_eq!(last_error(), "an older failure");
    assert!(handle.force_calls() > 0);
}

#[test]
fn hessian_out_overlapping_the_input_is_refused_without_a_call() {
    let mut system = System::octahedron();
    let handle = Handle::new();
    let dim = system.dim();
    let n = system.z.len();
    let central = RGPOT_HESSIAN_CENTRAL as i32;
    let mut used = RGPOT_HESSIAN_CENTRAL;

    let mut shared = vec![0.0; dim * dim];
    shared[dim * dim - dim..].copy_from_slice(&system.pos);
    let tail = unsafe { shared.as_mut_ptr().add(dim * dim - dim) };
    let input = unsafe {
        rgpot_force_input_create(n, tail, system.z.as_mut_ptr(), system.cell.as_mut_ptr())
    };
    let status = unsafe {
        rgpot_potential_hessian(
            handle.pot,
            &input,
            0.0,
            central,
            shared.as_mut_ptr(),
            &mut used,
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(last_error(), "hessian_out overlaps the positions tensor");

    let mut h = vec![0.0; dim * dim];
    h[dim..dim + 9].copy_from_slice(&system.cell);
    let cell = unsafe { h.as_mut_ptr().add(dim) };
    let input = unsafe {
        rgpot_force_input_create(n, system.pos.as_mut_ptr(), system.z.as_mut_ptr(), cell)
    };
    let status = unsafe {
        rgpot_potential_hessian(handle.pot, &input, 0.0, central, h.as_mut_ptr(), &mut used)
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(last_error(), "hessian_out overlaps the box_matrix tensor");

    let mut h = vec![0.0; dim * dim];
    let species = unsafe { h.as_mut_ptr().add(dim) }.cast::<i32>();
    let input = unsafe {
        rgpot_force_input_create(
            n,
            system.pos.as_mut_ptr(),
            species,
            system.cell.as_mut_ptr(),
        )
    };
    let status = unsafe {
        rgpot_potential_hessian(handle.pot, &input, 0.0, central, h.as_mut_ptr(), &mut used)
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(
        last_error(),
        "hessian_out overlaps the atomic_numbers tensor"
    );

    assert_eq!(handle.force_calls(), 0);
    assert_eq!(handle.counts().hessian_force_calls, 0);
}

#[test]
fn an_atom_count_whose_derivatives_overflow_is_refused_without_a_read() {
    let mut system = System::octahedron();
    let handle = Handle::new();
    let dim = system.dim();
    let mut h = vec![0.0; dim * dim];
    let v = vec![1.0; dim];
    let mut hv = vec![0.0; dim];
    let mut used = RGPOT_HESSIAN_CENTRAL;
    let wraps = usize::MAX / 3 + 1;
    let past_isize = 357_913_942;
    for n in [wraps, 1 << 31, past_isize] {
        let input = unsafe {
            rgpot_force_input_create(
                n,
                system.pos.as_mut_ptr(),
                system.z.as_mut_ptr(),
                system.cell.as_mut_ptr(),
            )
        };
        let status = unsafe {
            rgpot_potential_hessian(
                handle.pot,
                &input,
                0.0,
                RGPOT_HESSIAN_CENTRAL as i32,
                h.as_mut_ptr(),
                &mut used,
            )
        };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert_eq!(
            last_error(),
            "rgpot_potential_hessian: positions tensor is NULL or too large"
        );
    }
    for n in [wraps, isize::MAX as usize / 24 + 1] {
        let input = unsafe {
            rgpot_force_input_create(
                n,
                system.pos.as_mut_ptr(),
                system.z.as_mut_ptr(),
                system.cell.as_mut_ptr(),
            )
        };
        let status = unsafe {
            rgpot_potential_hvp(
                handle.pot,
                &input,
                v.as_ptr(),
                0.0,
                RGPOT_HVP_CENTRAL as i32,
                hv.as_mut_ptr(),
            )
        };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert!(last_error().ends_with("positions tensor is NULL or too large"));
    }
    assert_eq!(handle.force_calls(), 0);
}

#[test]
fn a_central_hessian_that_overflows_is_refused() {
    let mut system = System::octahedron();
    let input = system.input();
    let x0 = Box::new(system.pos.clone());
    let pot =
        unsafe { rgpot_potential_new(cliff_force, &*x0 as *const Vec<f64> as *mut c_void, None) };
    assert!(!pot.is_null());
    let dim = system.dim();
    let mut h = vec![0.0; dim * dim];
    let mut used = RGPOT_HESSIAN_CENTRAL;
    let status = unsafe {
        rgpot_potential_hessian(
            pot,
            &input,
            0.0,
            RGPOT_HESSIAN_CENTRAL as i32,
            h.as_mut_ptr(),
            &mut used,
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(
        last_error(),
        "central Hessian: non-finite value at element 0"
    );
    let mut counts = rgpot_derivative_counts_t::default();
    assert_eq!(
        unsafe { rgpot_potential_derivative_counts(pot, &mut counts) },
        rgpot_status_t::RGPOT_SUCCESS
    );
    assert_eq!(counts.hessian_central, 0);
    assert_eq!(counts.hessian_force_calls, 2 * dim as u64);
    unsafe { rgpot_potential_free(pot) };
}

#[test]
fn reset_zeros_every_derivative_count() {
    let mut system = System::octahedron();
    let input = system.input();
    let v = direction(system.dim());
    let handle = Handle::new();
    unsafe {
        rgpot_potential_set_ad_hvp_callback(handle.pot, Some(ad_hvp));
        rgpot_potential_set_hessian_callback(handle.pot, Some(analytic_hessian));
    }
    handle.try_hvp(&input, &v, RGPOT_HVP_AUTO as i32).unwrap();
    handle.try_hvp(&input, &v, RGPOT_HVP_AD as i32).unwrap();
    handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_ANALYTIC as i32)
        .unwrap();
    handle
        .try_hessian(&input, 0.0, RGPOT_HESSIAN_CENTRAL as i32)
        .unwrap();
    let before = handle.counts();
    assert_eq!(
        before,
        rgpot_derivative_counts_t {
            hvp_forward: 1,
            hvp_central: 0,
            hvp_analytic: 0,
            hvp_ad: 1,
            hessian_central: 1,
            hessian_analytic: 1,
            hessian_ad: 0,
            hessian_force_calls: 36,
        }
    );
    unsafe { rgpot_potential_reset_hvp_counts(handle.pot) };
    assert_eq!(handle.counts(), rgpot_derivative_counts_t::default());
    assert_eq!(handle.hvp_counts(), rgpot_hvp_counts_t::default());
    // The centre survives the reset, as for the other counters.
    handle
        .try_hvp(&input, &v, RGPOT_HVP_FORWARD as i32)
        .unwrap();
    assert_eq!(handle.hvp_counts().centre_reused, 1);
}

#[test]
fn a_zero_direction_counts_under_its_resolved_scheme() {
    let mut system = System::octahedron();
    let input = system.input();
    let handle = Handle::new();
    unsafe { rgpot_potential_set_ad_hvp_callback(handle.pot, Some(ad_hvp)) };
    let zero = vec![0.0; system.dim()];
    let (hv, used) = handle
        .try_hvp(&input, &zero, RGPOT_HVP_EXACT as i32)
        .unwrap();
    assert_eq!(used, RGPOT_HVP_AD);
    assert!(hv.iter().all(|x| *x == 0.0));
    assert_eq!(handle.counts().hvp_ad, 1);
    assert_eq!(Handle::calls(&handle.morse.ad_calls), 0);
    assert_eq!(handle.force_calls(), 0);
}
