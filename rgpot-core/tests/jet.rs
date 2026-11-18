// MIT License
// Copyright 2023--present rgpot developers

//! Taylor jets through the C ABI and the handle, against a bowl whose jet
//! is exact in binary64.
//!
//! The bowl is `V(x) = |x|^2`. Along `x + t v` its coefficients are
//! `|x|^2`, `2 x.v`, `|v|^2` and zeros, and with dyadic positions and
//! directions every one of them is a binary64 number, so the enclosures are
//! points and the round trip can be compared bit for bit.

use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use rgpot_core::c_api::jet::{rgpot_potential_set_taylor_jet, rgpot_potential_taylor_jet};
use rgpot_core::c_api::potential::{rgpot_potential_free, rgpot_potential_new};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::jet::{rgpot_interval_t, serve_taylor_jet, JetError, TaylorJet};
use rgpot_core::potential::rgpot_potential_t;
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

struct Bowl;

impl TaylorJet for Bowl {
    fn taylor_jet(
        &self,
        positions: &[f64],
        direction: &[f64],
        order: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetError> {
        if positions.len() != direction.len() {
            return Err(JetError::Shape("bowl direction length"));
        }
        let mut c = vec![0.0; order + 1];
        c[0] = positions.iter().map(|x| x * x).sum();
        if order >= 1 {
            c[1] = positions
                .iter()
                .zip(direction)
                .map(|(x, v)| 2.0 * x * v)
                .sum();
        }
        if order >= 2 {
            c[2] = direction.iter().map(|v| v * v).sum();
        }
        Ok(c.into_iter()
            .map(|c| rgpot_interval_t { lo: c, hi: c })
            .collect())
    }
}

struct Undecided;

impl TaylorJet for Undecided {
    fn taylor_jet(
        &self,
        _: &[f64],
        _: &[f64],
        _: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetError> {
        Err(JetError::Branch {
            test: "r < rcut",
            lo: 2.5,
            hi: 3.5,
        })
    }
}

struct Pole;

impl TaylorJet for Pole {
    fn taylor_jet(
        &self,
        _: &[f64],
        _: &[f64],
        _: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetError> {
        Err(JetError::Domain {
            function: "sqrt",
            lo: -1.0,
            hi: 0.0,
        })
    }
}

struct Opaque;

impl TaylorJet for Opaque {
    fn taylor_jet(
        &self,
        _: &[f64],
        _: &[f64],
        _: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetError> {
        Err(JetError::NoJet)
    }
}

unsafe extern "C" fn bowl_jet(
    _: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    unsafe { serve_taylor_jet(&Bowl, input, direction, order, lower, upper) }
}

static BOWL_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn counted_bowl_jet(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    BOWL_CALLS.fetch_add(1, Ordering::SeqCst);
    unsafe { bowl_jet(user_data, input, direction, order, lower, upper) }
}

unsafe extern "C" fn undecided_jet(
    _: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    unsafe { serve_taylor_jet(&Undecided, input, direction, order, lower, upper) }
}

unsafe extern "C" fn pole_jet(
    _: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    unsafe { serve_taylor_jet(&Pole, input, direction, order, lower, upper) }
}

unsafe extern "C" fn opaque_jet(
    _: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    unsafe { serve_taylor_jet(&Opaque, input, direction, order, lower, upper) }
}

/// Writes `[1, 0]` for every coefficient.
unsafe extern "C" fn reversed_jet(
    _: *mut c_void,
    _: *const rgpot_force_input_t,
    _: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    for k in 0..=order {
        unsafe {
            *lower.add(k) = 1.0;
            *upper.add(k) = 0.0;
        }
    }
    rgpot_status_t::RGPOT_SUCCESS
}

unsafe extern "C" fn no_force(
    _: *mut c_void,
    _: *const rgpot_force_input_t,
    _: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    rgpot_status_t::RGPOT_INTERNAL_ERROR
}

struct Frame {
    pos: Vec<f64>,
    z: Vec<i32>,
    cell: [f64; 9],
    pot: *mut rgpot_potential_t,
}

impl Frame {
    fn new() -> Self {
        Self {
            pos: vec![0.5, -1.25, 2.0, 0.75, 0.0, -0.5],
            z: vec![1, 1],
            cell: [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            pot: unsafe { rgpot_potential_new(no_force, std::ptr::null_mut(), None) },
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

    fn c_jet(
        &mut self,
        direction: &[f64],
        order: usize,
    ) -> Result<Vec<(f64, f64)>, rgpot_status_t> {
        let mut input = self.input();
        let mut lower = vec![f64::NAN; order + 1];
        let mut upper = vec![f64::NAN; order + 1];
        let status = unsafe {
            rgpot_potential_taylor_jet(
                self.pot,
                &input,
                direction.as_ptr(),
                order,
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
            )
        };
        unsafe { rgpot_force_input_free(&mut input) };
        if status == rgpot_status_t::RGPOT_SUCCESS {
            Ok(lower.into_iter().zip(upper).collect())
        } else {
            Err(status)
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free(self.pot) };
    }
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgpot_last_error()) }
        .to_str()
        .unwrap()
        .to_owned()
}

const DIRECTION: [f64; 6] = [1.0, 0.5, -0.25, 0.0, 2.0, 1.5];

#[test]
fn a_handle_without_a_jet_answers_no_jet() {
    let mut frame = Frame::new();
    assert_eq!(
        frame.c_jet(&DIRECTION, 4),
        Err(rgpot_status_t::RGPOT_NO_JET)
    );
    let mut input = frame.input();
    let failure = unsafe { (*frame.pot).taylor_jet(&input, &DIRECTION, 4) }.unwrap_err();
    unsafe { rgpot_force_input_free(&mut input) };
    assert_eq!(failure.status, rgpot_status_t::RGPOT_NO_JET);
    assert!(!unsafe { &*frame.pot }.has_taylor_jet());
}

#[test]
fn the_bowl_jet_round_trips_exactly() {
    let mut frame = Frame::new();
    let status = unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(bowl_jet)) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert!(unsafe { &*frame.pot }.has_taylor_jet());
    let x = &frame.pos;
    let c0: f64 = x.iter().map(|x| x * x).sum();
    let c1: f64 = x.iter().zip(&DIRECTION).map(|(x, v)| 2.0 * x * v).sum();
    let c2: f64 = DIRECTION.iter().map(|v| v * v).sum();
    assert_eq!((c0, c1, c2), (6.625, -2.75, 7.5625));
    let jet = frame.c_jet(&DIRECTION, 5).expect("bowl jet");
    assert_eq!(
        jet,
        vec![
            (c0, c0),
            (c1, c1),
            (c2, c2),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0)
        ]
    );
    assert_eq!(frame.c_jet(&DIRECTION, 0).expect("order 0"), vec![(c0, c0)]);
}

#[test]
fn clearing_the_jet_restores_no_jet() {
    let mut frame = Frame::new();
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(bowl_jet)) };
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, None) };
    assert_eq!(
        frame.c_jet(&DIRECTION, 2),
        Err(rgpot_status_t::RGPOT_NO_JET)
    );
}

#[test]
fn refusals_keep_their_status_and_their_enclosure() {
    let mut frame = Frame::new();
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(undecided_jet)) };
    assert_eq!(
        frame.c_jet(&DIRECTION, 3),
        Err(rgpot_status_t::RGPOT_JET_BRANCH)
    );
    assert_eq!(
        last_error(),
        "branch r < rcut: [2.5e0, 3.5e0] contains the threshold"
    );
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(pole_jet)) };
    assert_eq!(
        frame.c_jet(&DIRECTION, 3),
        Err(rgpot_status_t::RGPOT_JET_DOMAIN)
    );
    assert_eq!(last_error(), "sqrt at [-1e0, 0e0] is outside its domain");
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(opaque_jet)) };
    assert_eq!(
        frame.c_jet(&DIRECTION, 3),
        Err(rgpot_status_t::RGPOT_NO_JET)
    );

    let mut input = frame.input();
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(pole_jet)) };
    let failure = unsafe { (*frame.pot).taylor_jet(&input, &DIRECTION, 3) }.unwrap_err();
    unsafe { rgpot_force_input_free(&mut input) };
    assert_eq!(failure.status, rgpot_status_t::RGPOT_JET_DOMAIN);
    assert_eq!(failure.message, "sqrt at [-1e0, 0e0] is outside its domain");
}

#[test]
fn a_reversed_enclosure_from_the_callback_is_refused() {
    let mut frame = Frame::new();
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(reversed_jet)) };
    assert_eq!(
        frame.c_jet(&DIRECTION, 2),
        Err(rgpot_status_t::RGPOT_INVALID_PARAMETER)
    );
    assert_eq!(
        last_error(),
        "jet callback returned [1e0, 0e0] for coefficient 0"
    );
}

#[test]
fn malformed_requests_are_refused_before_the_callback() {
    let mut frame = Frame::new();
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(bowl_jet)) };
    let mut nan = DIRECTION;
    nan[3] = f64::NAN;
    assert_eq!(
        frame.c_jet(&nan, 2),
        Err(rgpot_status_t::RGPOT_INVALID_PARAMETER)
    );
    assert_eq!(last_error(), "direction: non-finite value at element 3");

    let mut input = frame.input();
    let failure = unsafe { (*frame.pot).taylor_jet(&input, &DIRECTION[..5], 2) }.unwrap_err();
    assert_eq!(failure.status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(failure.message, "direction: expected 6 elements, got 5");

    let mut slot = [0.0; 1];
    let status = unsafe {
        rgpot_potential_taylor_jet(
            frame.pot,
            &input,
            DIRECTION.as_ptr(),
            usize::MAX,
            slot.as_mut_ptr(),
            slot.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_BUFFER_SIZE_ERROR);

    let status = unsafe {
        rgpot_potential_taylor_jet(
            frame.pot,
            &input,
            DIRECTION.as_ptr(),
            0,
            std::ptr::null_mut(),
            slot.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    let status = unsafe {
        rgpot_potential_taylor_jet(
            std::ptr::null(),
            &input,
            DIRECTION.as_ptr(),
            0,
            slot.as_mut_ptr(),
            slot.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    unsafe { rgpot_force_input_free(&mut input) };
    let status = unsafe { rgpot_potential_set_taylor_jet(std::ptr::null(), Some(bowl_jet)) };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
}

/// The bowl jet through order 2 is `[6.625, -2.75, 7.5625]`; the cells it
/// does not own keep the sentinel `9.0`.
const BOWL: [f64; 3] = [6.625, -2.75, 7.5625];
const SENTINEL: f64 = 9.0;

#[test]
fn the_handle_refuses_overlapping_outputs_before_the_callback() {
    let mut frame = Frame::new();
    unsafe { rgpot_potential_set_taylor_jet(frame.pot, Some(counted_bowl_jet)) };
    BOWL_CALLS.store(0, Ordering::SeqCst);
    let mut input = frame.input();
    let mut buffer = [SENTINEL; 6];
    let p = buffer.as_mut_ptr();
    let call = |lower: *mut f64, upper: *mut f64| unsafe {
        rgpot_potential_taylor_jet(frame.pot, &input, DIRECTION.as_ptr(), 2, lower, upper)
    };

    for (lower, upper) in [(p, p), (p, unsafe { p.add(2) }), (unsafe { p.add(2) }, p)] {
        assert_eq!(call(lower, upper), rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert_eq!(
            last_error(),
            "rgpot_potential_taylor_jet: lower and upper overlap"
        );
    }
    let wrapping = (usize::MAX - 8) as *mut f64;
    assert_eq!(call(wrapping, p), rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(
        last_error(),
        "rgpot_potential_taylor_jet: lower or upper wraps the address space"
    );
    assert_eq!(BOWL_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(buffer, [SENTINEL; 6]);

    let p = buffer.as_mut_ptr();
    let status = unsafe {
        rgpot_potential_taylor_jet(frame.pot, &input, DIRECTION.as_ptr(), 2, p, p.add(3))
    };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(BOWL_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        buffer,
        [BOWL[0], BOWL[1], BOWL[2], BOWL[0], BOWL[1], BOWL[2]]
    );
    unsafe { rgpot_force_input_free(&mut input) };
}

#[test]
fn the_callback_adapter_refuses_overlapping_outputs_before_the_jet() {
    let mut frame = Frame::new();
    let mut input = frame.input();
    let mut buffer = [SENTINEL; 7];
    let p = buffer.as_mut_ptr();
    let bowl = |lower: *mut f64, upper: *mut f64| unsafe {
        serve_taylor_jet(&Bowl, &input, DIRECTION.as_ptr(), 2, lower, upper)
    };

    for (lower, upper) in [(p, p), (p, unsafe { p.add(1) }), (unsafe { p.add(2) }, p)] {
        assert_eq!(bowl(lower, upper), rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert_eq!(last_error(), "Taylor jet: lower and upper overlap");
    }
    let wrapping = (usize::MAX - 8) as *mut f64;
    assert_eq!(bowl(p, wrapping), rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(
        last_error(),
        "Taylor jet: lower or upper wraps the address space"
    );
    assert_eq!(buffer, [SENTINEL; 7]);

    let status =
        unsafe { serve_taylor_jet(&Bowl, &input, DIRECTION.as_ptr(), 2, p.add(4), p.add(1)) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    assert_eq!(
        buffer,
        [SENTINEL, BOWL[0], BOWL[1], BOWL[2], BOWL[0], BOWL[1], BOWL[2]]
    );
    unsafe { rgpot_force_input_free(&mut input) };
}
