// MIT License
// Copyright 2023--present rgpot developers

//! Hessian-vector products through the C ABI, against a Morse cluster
//! whose Hessian is known in closed form.
//!
//! The model is a periodic Morse solid: each pair interacts through its
//! minimum image in an orthorhombic cell, and its well depth is weighted by
//! the atomic numbers. The cluster straddles a cell corner, so the forces
//! depend on the positions, the species and the cell, and a stale centre
//! of any of the three gives a wrong product.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use rgpot_core::c_api::hvp::{
    rgpot_potential_hvp, rgpot_potential_hvp_centre, rgpot_potential_hvp_counts,
    rgpot_potential_hvp_invalidate, rgpot_potential_reset_hvp_counts,
    rgpot_potential_set_hvp_callback,
};
use rgpot_core::c_api::potential::{
    rgpot_potential_calculate, rgpot_potential_free, rgpot_potential_new,
};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::hvp::{rgpot_hvp_counts_t, rgpot_hvp_scheme_t, HvpError, RGPOT_HVP_DEFAULT_STEP};
use rgpot_core::potential::rgpot_potential_t;
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::tensor::{rgpot_tensor_free, rgpot_tensor_owned_cpu_f64_2d};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const AUTO: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_AUTO as i32;
const FORWARD: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD as i32;
const CENTRAL: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_CENTRAL as i32;
const ANALYTIC: i32 = rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC as i32;

struct Morse {
    de: f64,
    a: f64,
    re: f64,
    force_calls: AtomicUsize,
    hvp_calls: AtomicUsize,
}

impl Morse {
    fn platinum() -> Self {
        Self {
            de: 0.7102,
            a: 1.6047,
            re: 2.8970,
            force_calls: AtomicUsize::new(0),
            hvp_calls: AtomicUsize::new(0),
        }
    }

    /// Pair energy and its first two radial derivatives.
    fn pair(&self, de: f64, r: f64) -> (f64, f64, f64) {
        let e = (-self.a * (r - self.re)).exp();
        let phi = de * (1.0 - e) * (1.0 - e) - de;
        let dphi = 2.0 * de * self.a * e * (1.0 - e);
        let d2phi = 2.0 * de * self.a * self.a * e * (2.0 * e - 1.0);
        (phi, dphi, d2phi)
    }
}

/// A geometry as the callbacks see it.
struct Frame {
    pos: Vec<f64>,
    z: Option<Vec<i32>>,
    cell: Option<[f64; 3]>,
}

impl Frame {
    unsafe fn read(input: &rgpot_force_input_t) -> Self {
        let n = unsafe { input.n_atoms() }.unwrap();
        let pos = unsafe {
            std::slice::from_raw_parts((*input.positions).dl_tensor.data as *const f64, 3 * n)
        }
        .to_vec();
        let z = (!input.atomic_numbers.is_null()).then(|| {
            unsafe {
                std::slice::from_raw_parts((*input.atomic_numbers).dl_tensor.data as *const i32, n)
            }
            .to_vec()
        });
        let cell = (!input.box_matrix.is_null()).then(|| {
            let b = unsafe {
                std::slice::from_raw_parts((*input.box_matrix).dl_tensor.data as *const f64, 9)
            };
            [b[0], b[4], b[8]]
        });
        Self { pos, z, cell }
    }

    fn n(&self) -> usize {
        self.pos.len() / 3
    }

    fn weight(&self, i: usize) -> f64 {
        self.z
            .as_ref()
            .map_or(1.0, |z| 1.0 + 0.002 * f64::from(z[i]))
    }

    /// Minimum-image separation `r_i - r_j`.
    fn separation(&self, i: usize, j: usize) -> [f64; 3] {
        let mut d = [0.0; 3];
        for k in 0..3 {
            d[k] = self.pos[3 * i + k] - self.pos[3 * j + k];
            if let Some(cell) = self.cell {
                d[k] -= cell[k] * (d[k] / cell[k]).round();
            }
        }
        d
    }

    fn for_pairs(&self, mut f: impl FnMut(usize, usize, [f64; 3], f64, f64)) {
        for i in 0..self.n() {
            for j in (i + 1)..self.n() {
                let d = self.separation(i, j);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                f(i, j, d, r, self.weight(i) * self.weight(j));
            }
        }
    }
}

impl Morse {
    fn energy_forces(&self, frame: &Frame) -> (f64, Vec<f64>) {
        let mut energy = 0.0;
        let mut forces = vec![0.0; frame.pos.len()];
        frame.for_pairs(|i, j, d, r, w| {
            let (phi, dphi, _) = self.pair(self.de * w, r);
            energy += phi;
            for k in 0..3 {
                let f = -dphi * d[k] / r;
                forces[3 * i + k] += f;
                forces[3 * j + k] -= f;
            }
        });
        (energy, forces)
    }

    /// `H v` from the pair blocks
    /// `K = phi'' d d^T + (phi' / r) (I - d d^T)`, `d` the unit separation.
    fn hessian_vector(&self, frame: &Frame, v: &[f64]) -> Vec<f64> {
        let mut hv = vec![0.0; v.len()];
        frame.for_pairs(|i, j, d, r, w| {
            let (_, dphi, d2phi) = self.pair(self.de * w, r);
            let u = [d[0] / r, d[1] / r, d[2] / r];
            let dv = [
                v[3 * i] - v[3 * j],
                v[3 * i + 1] - v[3 * j + 1],
                v[3 * i + 2] - v[3 * j + 2],
            ];
            let along = u[0] * dv[0] + u[1] * dv[1] + u[2] * dv[2];
            for k in 0..3 {
                let kv = d2phi * along * u[k] + dphi / r * (dv[k] - along * u[k]);
                hv[3 * i + k] += kv;
                hv[3 * j + k] -= kv;
            }
        });
        hv
    }
}

unsafe extern "C" fn morse_force(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.force_calls.fetch_add(1, Ordering::SeqCst);
    let frame = unsafe { Frame::read(&*input) };
    let (energy, forces) = morse.energy_forces(&frame);
    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), frame.n() as i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn morse_hvp(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    hv_out: *mut f64,
) -> rgpot_status_t {
    let morse = unsafe { &*(user_data as *const Morse) };
    morse.hvp_calls.fetch_add(1, Ordering::SeqCst);
    let frame = unsafe { Frame::read(&*input) };
    let dim = frame.pos.len();
    let v = unsafe { std::slice::from_raw_parts(direction, dim) };
    let hv = morse.hessian_vector(&frame, v);
    unsafe { std::slice::from_raw_parts_mut(hv_out, dim) }.copy_from_slice(&hv);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Six atoms on a slightly distorted octahedron of edge about 2.83 A,
/// centred on a corner of a 10 A cell and wrapped into it.
fn octahedron() -> Vec<f64> {
    let unwrapped = [
        2.00, 0.10, -0.05, //
        -1.95, -0.08, 0.12, //
        0.07, 2.03, 0.02, //
        -0.10, -2.05, -0.06, //
        0.04, -0.03, 1.98, //
        0.02, 0.09, -2.02,
    ];
    unwrapped.iter().map(|x: &f64| x.rem_euclid(10.0)).collect()
}

fn cell(edge: f64) -> [f64; 9] {
    [edge, 0.0, 0.0, 0.0, edge, 0.0, 0.0, 0.0, edge]
}

/// A deterministic direction with every component nonzero.
fn direction(dim: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The arrays an `rgpot_force_input_t` borrows, kept alive by the test.
struct System {
    pos: Vec<f64>,
    z: Vec<i32>,
    cell: [f64; 9],
}

impl System {
    fn octahedron() -> Self {
        Self {
            pos: octahedron(),
            z: vec![78, 78, 78, 46, 46, 78],
            cell: cell(10.0),
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

    fn frame(&self) -> Frame {
        Frame {
            pos: self.pos.clone(),
            z: Some(self.z.clone()),
            cell: Some([self.cell[0], self.cell[4], self.cell[8]]),
        }
    }
}

struct Handle {
    pot: *mut rgpot_potential_t,
    morse: Box<Morse>,
}

impl Handle {
    fn new() -> Self {
        let morse = Box::new(Morse::platinum());
        let pot = unsafe {
            rgpot_potential_new(morse_force, &*morse as *const Morse as *mut c_void, None)
        };
        assert!(!pot.is_null());
        Self { pot, morse }
    }

    fn with_analytic() -> Self {
        let handle = Self::new();
        let status = unsafe { rgpot_potential_set_hvp_callback(handle.pot, Some(morse_hvp)) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        handle
    }

    fn force_calls(&self) -> usize {
        self.morse.force_calls.load(Ordering::SeqCst)
    }

    fn counts(&self) -> rgpot_hvp_counts_t {
        let mut out = rgpot_hvp_counts_t::default();
        let status = unsafe { rgpot_potential_hvp_counts(self.pot, &mut out) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        out
    }

    fn hvp(&self, input: &rgpot_force_input_t, v: &[f64], step: f64, scheme: i32) -> Vec<f64> {
        let mut hv = vec![f64::NAN; v.len()];
        let status = unsafe {
            rgpot_potential_hvp(self.pot, input, v.as_ptr(), step, scheme, hv.as_mut_ptr())
        };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS, "{}", last_error());
        hv
    }

    fn try_hvp(
        &self,
        input: &rgpot_force_input_t,
        v: &[f64],
        step: f64,
        scheme: i32,
    ) -> rgpot_status_t {
        let mut hv = vec![0.0; v.len()];
        unsafe { rgpot_potential_hvp(self.pot, input, v.as_ptr(), step, scheme, hv.as_mut_ptr()) }
    }

    /// Forces from a plain `rgpot_potential_calculate`.
    fn forces(&self, input: &rgpot_force_input_t) -> Vec<f64> {
        let mut out = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(self.pot, input, &mut out) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        let n = unsafe { input.n_atoms() }.unwrap();
        let forces = unsafe {
            std::slice::from_raw_parts((*out.forces).dl_tensor.data as *const f64, 3 * n)
        }
        .to_vec();
        unsafe { rgpot_tensor_free(out.forces) };
        forces
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

/// `k` steps of Lanczos from `q1` with `apply` as the operator. Returns the
/// diagonal of the tridiagonal matrix.
fn lanczos(k: usize, q1: &[f64], mut apply: impl FnMut(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let norm = dot(q1, q1).sqrt();
    let mut q: Vec<f64> = q1.iter().map(|x| x / norm).collect();
    let mut q_prev = vec![0.0; q.len()];
    let mut beta = 0.0;
    let mut alphas = Vec::with_capacity(k);
    for _ in 0..k {
        let mut w = apply(&q);
        let alpha = dot(&w, &q);
        for ((wi, qi), pi) in w.iter_mut().zip(&q).zip(&q_prev) {
            *wi -= alpha * qi + beta * pi;
        }
        alphas.push(alpha);
        beta = dot(&w, &w).sqrt();
        q_prev = std::mem::replace(&mut q, w.iter().map(|x| x / beta).collect());
    }
    alphas
}

#[test]
fn a_lanczos_loop_of_k_products_costs_k_plus_one_force_calls() {
    let k = 8;
    let mut system = System::octahedron();
    let mut input = system.input();
    let q1 = direction(system.pos.len(), 7);
    let h = RGPOT_HVP_DEFAULT_STEP;

    let cached = Handle::new();
    let alphas = lanczos(k, &q1, |q| cached.hvp(&input, q, 0.0, FORWARD));
    assert_eq!(cached.force_calls(), k + 1);
    let counts = cached.counts();
    assert_eq!(counts.products, k as u64);
    assert_eq!(counts.force_calls, (k + 1) as u64);
    assert_eq!(counts.centre_reused, (k - 1) as u64);
    assert_eq!(counts.analytic, 0);

    // The same loop without a cached centre: each product evaluates the
    // centre and the displaced geometry.
    let naive = Handle::new();
    let reference = lanczos(k, &q1, |q| {
        let f0 = naive.forces(&input);
        let len = dot(q, q).sqrt();
        let mut moved = System {
            pos: system
                .pos
                .iter()
                .zip(q)
                .map(|(x, v)| x + h * v / len)
                .collect(),
            z: system.z.clone(),
            cell: system.cell,
        };
        let mut moved_input = moved.input();
        let fp = naive.forces(&moved_input);
        unsafe { rgpot_force_input_free(&mut moved_input) };
        f0.iter().zip(&fp).map(|(a, b)| (a - b) * len / h).collect()
    });
    assert_eq!(naive.force_calls(), 2 * k);
    for (a, b) in alphas.iter().zip(&reference) {
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs {b}");
    }

    let analytic = Handle::with_analytic();
    let exact = lanczos(k, &q1, |q| analytic.hvp(&input, q, 0.0, AUTO));
    assert_eq!(analytic.force_calls(), 0);
    assert_eq!(analytic.morse.hvp_calls.load(Ordering::SeqCst), k);
    assert_eq!(analytic.counts().analytic, k as u64);
    // The first Lanczos coefficient is the Rayleigh quotient of q1.
    assert!((alphas[0] - exact[0]).abs() < 1e-2 * exact[0].abs().max(1.0));

    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn the_centre_entry_shares_its_evaluation_with_the_products() {
    let k = 5;
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();

    let mut out = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { rgpot_potential_hvp_centre(handle.pot, &input, &mut out) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let (energy, forces) = handle.morse.energy_forces(&system.frame());
    assert_eq!(out.energy.to_bits(), energy.to_bits());
    let got = unsafe {
        std::slice::from_raw_parts((*out.forces).dl_tensor.data as *const f64, forces.len())
    };
    assert_eq!(got, &forces[..]);
    unsafe { rgpot_tensor_free(out.forces) };

    for seed in 0..k {
        handle.hvp(
            &input,
            &direction(system.pos.len(), seed as u64),
            0.0,
            FORWARD,
        );
    }
    assert_eq!(handle.force_calls(), k + 1);
    assert_eq!(handle.counts().centre_reused, k as u64);

    unsafe { rgpot_force_input_free(&mut input) };
}

/// Largest `|FD - analytic| / |analytic|` over a few directions, at each
/// step.
fn errors(scheme: i32, steps: &[f64]) -> Vec<f64> {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let frame = system.frame();
    let dim = system.pos.len();
    let out = steps
        .iter()
        .map(|&h| {
            (0..4)
                .map(|seed| {
                    let v = direction(dim, 100 + seed);
                    let exact = handle.morse.hessian_vector(&frame, &v);
                    distance(&handle.hvp(&input, &v, h, scheme), &exact)
                        / dot(&exact, &exact).sqrt()
                })
                .fold(0.0, f64::max)
        })
        .collect();
    unsafe { rgpot_force_input_free(&mut input) };
    out
}

/// `log2` of successive error ratios when the step halves.
fn observed_orders(errors: &[f64]) -> Vec<f64> {
    errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
}

#[test]
fn forward_differences_converge_at_first_order() {
    let steps = [8e-3, 4e-3, 2e-3, 1e-3];
    let errs = errors(FORWARD, &steps);
    for order in observed_orders(&errs) {
        assert!((order - 1.0).abs() < 0.1, "order {order}, errors {errs:?}");
    }
    assert!(errs[3] < 1e-2, "{errs:?}");
}

#[test]
fn central_differences_converge_at_second_order() {
    let steps = [4e-2, 2e-2, 1e-2, 5e-3];
    let errs = errors(CENTRAL, &steps);
    for order in observed_orders(&errs) {
        assert!((order - 2.0).abs() < 0.1, "order {order}, errors {errs:?}");
    }
    let at_default = errors(CENTRAL, &[RGPOT_HVP_DEFAULT_STEP])[0];
    assert!(at_default < 1e-5, "{at_default}");
    let forward_at_default = errors(FORWARD, &[RGPOT_HVP_DEFAULT_STEP])[0];
    assert!(
        at_default < 1e-2 * forward_at_default,
        "{at_default} {forward_at_default}"
    );
}

#[test]
fn central_differences_make_two_calls_and_ignore_the_centre() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let v = direction(system.pos.len(), 3);
    handle.hvp(&input, &v, 0.0, CENTRAL);
    handle.hvp(&input, &v, 0.0, CENTRAL);
    assert_eq!(handle.force_calls(), 4);
    assert_eq!(handle.counts().centre_reused, 0);
    unsafe { rgpot_force_input_free(&mut input) };
}

/// One forward product after `change`, checked against the analytic
/// product at the changed geometry. Returns the force calls it made.
fn calls_after(change: impl FnOnce(&mut System)) -> usize {
    let mut system = System::octahedron();
    let handle = Handle::new();
    let v = direction(system.pos.len(), 11);
    let mut input = system.input();
    handle.hvp(&input, &v, 1e-4, FORWARD);
    unsafe { rgpot_force_input_free(&mut input) };
    let before = handle.force_calls();

    change(&mut system);
    let mut input = system.input();
    let hv = handle.hvp(&input, &v, 1e-4, FORWARD);
    let exact = handle.morse.hessian_vector(&system.frame(), &v);
    assert!(
        distance(&hv, &exact) < 1e-2 * dot(&exact, &exact).sqrt(),
        "stale centre: {hv:?} vs {exact:?}"
    );
    unsafe { rgpot_force_input_free(&mut input) };
    handle.force_calls() - before
}

#[test]
fn an_unchanged_geometry_reuses_the_centre() {
    assert_eq!(calls_after(|_| {}), 1);
}

#[test]
fn moved_positions_invalidate_the_centre() {
    assert_eq!(calls_after(|s| s.pos[4] += 0.05), 2);
}

#[test]
fn a_one_ulp_move_invalidates_the_centre() {
    assert_eq!(
        calls_after(|s| s.pos[4] = f64::from_bits(s.pos[4].to_bits() + 1)),
        2
    );
}

#[test]
fn a_changed_cell_invalidates_the_centre() {
    assert_eq!(calls_after(|s| s.cell = cell(10.5)), 2);
}

#[test]
fn changed_species_invalidate_the_centre() {
    assert_eq!(calls_after(|s| s.z[0] = 46), 2);
}

#[test]
fn a_negative_zero_is_a_new_centre() {
    let mut system = System::octahedron();
    system.pos[0] = 0.0;
    let handle = Handle::new();
    let v = direction(system.pos.len(), 5);
    let mut input = system.input();
    let first = handle.hvp(&input, &v, 0.0, FORWARD);
    unsafe { rgpot_force_input_free(&mut input) };
    system.pos[0] = -0.0;
    let mut input = system.input();
    let second = handle.hvp(&input, &v, 0.0, FORWARD);
    unsafe { rgpot_force_input_free(&mut input) };
    assert_eq!(handle.force_calls(), 4);
    assert_eq!(first, second);
}

#[test]
fn explicit_invalidation_recomputes_the_centre() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let v = direction(system.pos.len(), 1);
    handle.hvp(&input, &v, 0.0, FORWARD);
    unsafe { rgpot_potential_hvp_invalidate(handle.pot) };
    handle.hvp(&input, &v, 0.0, FORWARD);
    assert_eq!(handle.force_calls(), 4);
    unsafe { rgpot_potential_hvp_invalidate(std::ptr::null()) };
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn the_analytic_scheme_matches_the_closed_form_and_can_be_bypassed() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::with_analytic();
    let v = direction(system.pos.len(), 9);
    let exact = handle.morse.hessian_vector(&system.frame(), &v);
    assert_eq!(handle.hvp(&input, &v, 0.0, ANALYTIC), exact);
    assert_eq!(handle.hvp(&input, &v, 0.0, AUTO), exact);
    assert_eq!(handle.force_calls(), 0);

    let central = handle.hvp(&input, &v, 0.0, CENTRAL);
    assert!(distance(&central, &exact) < 1e-5);
    assert_eq!(handle.force_calls(), 2);

    let status = unsafe { rgpot_potential_set_hvp_callback(handle.pot, None) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    handle.hvp(&input, &v, 0.0, AUTO);
    assert_eq!(handle.force_calls(), 4);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn products_are_linear_in_the_direction() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let v = direction(system.pos.len(), 2);
    let hv = handle.hvp(&input, &v, 0.0, FORWARD);
    let scaled: Vec<f64> = v.iter().map(|x| 1e3 * x).collect();
    let hv_scaled = handle.hvp(&input, &scaled, 0.0, FORWARD);
    for (a, b) in hv.iter().zip(&hv_scaled) {
        assert!((1e3 * a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} {b}");
    }
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn a_zero_direction_is_a_zero_product_without_calls() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let v = vec![0.0; system.pos.len()];
    assert!(handle
        .hvp(&input, &v, 0.0, FORWARD)
        .iter()
        .all(|&x| x == 0.0));
    assert_eq!(handle.force_calls(), 0);
    assert_eq!(handle.counts().products, 1);
    unsafe { rgpot_potential_reset_hvp_counts(handle.pot) };
    assert_eq!(handle.counts(), rgpot_hvp_counts_t::default());
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn malformed_requests_are_refused_before_any_call() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let v = direction(system.pos.len(), 4);
    let invalid = rgpot_status_t::RGPOT_INVALID_PARAMETER;

    assert_eq!(handle.try_hvp(&input, &v, 0.0, 7), invalid);
    assert!(
        last_error().contains("unknown scheme 7"),
        "{}",
        last_error()
    );
    assert_eq!(handle.try_hvp(&input, &v, 0.0, ANALYTIC), invalid);
    assert!(
        last_error().contains("no Hessian-vector callback"),
        "{}",
        last_error()
    );
    for step in [-1e-3, f64::NAN, f64::INFINITY] {
        assert_eq!(handle.try_hvp(&input, &v, step, FORWARD), invalid);
    }
    let mut bad = v.clone();
    bad[2] = f64::NAN;
    assert_eq!(handle.try_hvp(&input, &bad, 0.0, FORWARD), invalid);

    let mut hv = v.clone();
    let p = hv.as_mut_ptr();
    let status = unsafe { rgpot_potential_hvp(handle.pot, &input, p, 0.0, FORWARD, p) };
    assert_eq!(status, invalid);
    assert!(last_error().contains("overlap"));
    let status =
        unsafe { rgpot_potential_hvp(handle.pot, &input, std::ptr::null(), 0.0, FORWARD, p) };
    assert_eq!(status, invalid);
    let status =
        unsafe { rgpot_potential_hvp(std::ptr::null(), &input, v.as_ptr(), 0.0, FORWARD, p) };
    assert_eq!(status, invalid);
    let status = unsafe { rgpot_potential_hvp_centre(handle.pot, &input, std::ptr::null_mut()) };
    assert_eq!(status, invalid);
    let status = unsafe { rgpot_potential_hvp_counts(handle.pot, std::ptr::null_mut()) };
    assert_eq!(status, invalid);
    let status = unsafe { rgpot_potential_set_hvp_callback(std::ptr::null(), None) };
    assert_eq!(status, invalid);

    assert_eq!(handle.force_calls(), 0);
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn the_rust_api_checks_lengths() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let handle = Handle::new();
    let v = direction(system.pos.len() - 3, 4);
    let mut hv = vec![0.0; system.pos.len()];
    let err = unsafe {
        (*handle.pot).hessian_vector(
            &input,
            &v,
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD,
            &mut hv,
        )
    }
    .unwrap_err();
    assert!(
        matches!(&err, HvpError::Invalid(why) if why.contains("direction")),
        "{err}"
    );
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
    unsafe { rgpot_force_input_free(&mut input) };
}

static FAIL_AFTER: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn failing_force(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    if FAIL_AFTER.fetch_sub(1, Ordering::SeqCst) == 0 {
        FAIL_AFTER.store(0, Ordering::SeqCst);
        return rgpot_status_t::RGPOT_RPC_ERROR;
    }
    unsafe { morse_force(user_data, input, output) }
}

#[test]
fn a_failed_force_call_returns_its_status_and_caches_nothing() {
    let mut system = System::octahedron();
    let mut input = system.input();
    let morse = Box::new(Morse::platinum());
    let pot =
        unsafe { rgpot_potential_new(failing_force, &*morse as *const Morse as *mut c_void, None) };
    let v = direction(system.pos.len(), 6);
    let mut hv = vec![7.0; v.len()];

    FAIL_AFTER.store(0, Ordering::SeqCst);
    let status =
        unsafe { rgpot_potential_hvp(pot, &input, v.as_ptr(), 0.0, FORWARD, hv.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    assert!(hv.iter().all(|&x| x == 7.0));

    // The centre succeeds and the displaced call fails: the centre is kept.
    FAIL_AFTER.store(1, Ordering::SeqCst);
    let status =
        unsafe { rgpot_potential_hvp(pot, &input, v.as_ptr(), 0.0, FORWARD, hv.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    FAIL_AFTER.store(usize::MAX, Ordering::SeqCst);
    let status =
        unsafe { rgpot_potential_hvp(pot, &input, v.as_ptr(), 0.0, FORWARD, hv.as_mut_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let mut counts = rgpot_hvp_counts_t::default();
    unsafe { rgpot_potential_hvp_counts(pot, &mut counts) };
    assert_eq!(counts.products, 1);
    assert_eq!(counts.centre_reused, 1);
    assert_eq!(counts.force_calls, 4);

    unsafe {
        rgpot_potential_free(pot);
        rgpot_force_input_free(&mut input);
    }
}
