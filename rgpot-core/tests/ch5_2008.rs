// MIT License
// Copyright 2023--present rgpot developers

//! The CH5 2008 jet against an independent reference.
//!
//! `data/ch5_2008/reference.txt` is written by `data/ch5_2008/generate.py`.
//! For eight geometries and directions near the H + CH4 saddle it holds,
//! from a transcription of `POT` of `ch52008.f` into mpmath at 60 digits:
//!
//! - every coefficient `c_0 ... c_12` of `V(x + t v)` as the binary64
//!   bracket of its exact value, from power series checked against
//!   `mpmath.taylor`;
//! - the energy `E`, the slope `g . v` and the curvature `v . H v` as
//!   binary64 brackets, from partial derivatives of the scalar energy taken
//!   by `mpmath.diff` on the 18 coordinates.
//!
//! The exact value lies somewhere in its bracket, so an enclosure that only
//! meets the bracket can still exclude it: every coefficient must contain its
//! bracket, and `c_0`, `c_1` and `2 c_2` must contain those of `E`, `g . v`
//! and `v . H v`. The widest relative enclosure is
//! reported, not bounded.
//!
//! The file also holds the energy, gradient and product of the compiled
//! source (libcbe). Its gradient and product convert with the truncated
//! factor `0.0201723` where its energy uses `0.03812 * 0.52918`, so the
//! comparison with them is ignored until the potlibc wrapper fix lands and
//! the file is regenerated from the fixed library.

use std::ffi::CStr;

use rgpot_core::c_api::jet::{rgpot_potential_set_taylor_jet, rgpot_potential_taylor_jet};
use rgpot_core::c_api::potential::{rgpot_potential_free, rgpot_potential_new};
use rgpot_core::c_api::types::{rgpot_force_input_create, rgpot_force_input_free};
use rgpot_core::ch5_2008::{rgpot_ch5_2008_taylor_jet, Ch52008, CH5_2008_ATOMIC_NUMBERS};
use rgpot_core::jet::{JetError, TaylorJet};
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::taylor::Interval;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// The H + CH4 saddle of the surface, Bohr.
const SADDLE: [f64; 18] = [
    2.5078986937601746,
    3.0439878740728727,
    2.603957609783651,
    1.1100980488026544,
    1.3613545004038685,
    1.1486133528262976,
    0.013880929393421192,
    0.2086334261632169,
    2.457516169583444,
    2.4611712350631576,
    0.23586419033998002,
    0.07550700261319757,
    -0.09665148160819977,
    2.4937031206965017,
    -0.0782634154328256,
    3.3853544918540113,
    4.100244654765713,
    3.5175358525837166,
];

fn mixed() -> Vec<f64> {
    (0..18)
        .map(|i| (((7 * i) % 11) as f64 - 4.5) / 8.0)
        .collect()
}

const REFERENCE: &str = include_str!("data/ch5_2008/reference.txt");

struct Case {
    label: String,
    x: Vec<f64>,
    v: Vec<f64>,
    brackets: Vec<(f64, f64)>,
    energy: (f64, f64),
    slope: (f64, f64),
    curvature: (f64, f64),
    libcbe_energy: f64,
    libcbe_gradient: Vec<f64>,
    libcbe_hvp: Vec<f64>,
    ops: f64,
    scale: f64,
}

fn numbers(fields: &[&str]) -> Vec<f64> {
    fields
        .iter()
        .map(|f| f.parse().expect("reference number"))
        .collect()
}

fn reference() -> (usize, Vec<Case>) {
    let mut order = None;
    let mut cases: Vec<Case> = Vec::new();
    for line in REFERENCE.lines().filter(|l| !l.starts_with('#')) {
        if let Some(label) = line.strip_prefix("case ") {
            cases.push(Case {
                label: label.to_owned(),
                x: Vec::new(),
                v: Vec::new(),
                brackets: Vec::new(),
                energy: (f64::NAN, f64::NAN),
                slope: (f64::NAN, f64::NAN),
                curvature: (f64::NAN, f64::NAN),
                libcbe_energy: f64::NAN,
                libcbe_gradient: Vec::new(),
                libcbe_hvp: Vec::new(),
                ops: f64::NAN,
                scale: f64::NAN,
            });
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields[0] == "order" {
            order = Some(fields[1].parse().expect("order"));
            continue;
        }
        let case = cases.last_mut().expect("a field before the first case");
        let values = numbers(&fields[1..]);
        match fields[0] {
            "x" => case.x = values,
            "v" => case.v = values,
            "c" => {
                assert_eq!(values[0] as usize, case.brackets.len(), "{}", case.label);
                case.brackets.push((values[1], values[2]));
            }
            "energy" => case.energy = (values[0], values[1]),
            "slope" => case.slope = (values[0], values[1]),
            "curvature" => case.curvature = (values[0], values[1]),
            "libcbe_energy" => case.libcbe_energy = values[0],
            "libcbe_gradient" => case.libcbe_gradient = values,
            "libcbe_hvp" => case.libcbe_hvp = values,
            "libcbe_hvp_roundoff" => (case.ops, case.scale) = (values[0], values[1]),
            other => panic!("unknown reference field {other}"),
        }
    }
    let order = order.expect("reference order");
    for case in &cases {
        assert_eq!(case.x.len(), 18, "{}", case.label);
        assert_eq!(case.v.len(), 18, "{}", case.label);
        assert_eq!(case.brackets.len(), order + 1, "{}", case.label);
        for (down, up) in [case.energy, case.slope, case.curvature] {
            assert!(down <= up, "{}", case.label);
        }
        assert_eq!(case.libcbe_gradient.len(), 18, "{}", case.label);
        assert_eq!(case.libcbe_hvp.len(), 18, "{}", case.label);
        assert!(case.libcbe_energy.is_finite() && case.ops > 0.0 && case.scale > 0.0);
    }
    assert_eq!(cases.len(), 8);
    (order, cases)
}

fn gap(got: Interval, want: f64) -> f64 {
    if want < got.inf() {
        got.inf() - want
    } else if want > got.sup() {
        want - got.sup()
    } else {
        0.0
    }
}

fn interval(c: &rgpot_core::jet::rgpot_interval_t) -> Interval {
    Interval::new(c.lo, c.hi)
}

fn point(a: f64) -> Interval {
    Interval::new(a, a)
}

fn contains(got: Interval, (down, up): (f64, f64)) -> bool {
    got.inf() <= down && up <= got.sup()
}

#[test]
fn every_coefficient_encloses_the_independent_reference() {
    let (order, cases) = reference();
    for case in cases {
        let jet = Ch52008
            .taylor_jet(&case.x, &case.v, order)
            .expect("CH5 jet");
        assert_eq!(jet.len(), order + 1);
        for (k, (c, &(down, up))) in jet.iter().zip(&case.brackets).enumerate() {
            assert!(down <= up, "{} c_{k} bracket", case.label);
            assert!(
                c.lo.is_finite() && c.hi.is_finite(),
                "{} c_{k} [{:e}, {:e}]",
                case.label,
                c.lo,
                c.hi
            );
            assert!(
                c.lo <= down && up <= c.hi,
                "{} c_{k}: [{:.17e}, {:.17e}] does not contain [{:.17e}, {:.17e}]",
                case.label,
                c.lo,
                c.hi,
                down,
                up
            );
        }
        let widest = jet
            .iter()
            .map(|c| (c.hi - c.lo) / c.lo.abs().max(c.hi.abs()).max(1.0))
            .fold(0.0, f64::max);
        eprintln!("{}: widest relative enclosure {widest:.3e}", case.label);
    }
}

#[test]
fn the_low_orders_contain_the_independent_energy_slope_and_curvature() {
    let (_, cases) = reference();
    for case in cases {
        let jet = Ch52008.taylor_jet(&case.x, &case.v, 2).expect("CH5 jet");
        let pairs = [
            ("E", interval(&jet[0]), case.energy),
            ("g . v", interval(&jet[1]), case.slope),
            ("v . H v", point(2.0) * interval(&jet[2]), case.curvature),
        ];
        for (name, got, (down, up)) in pairs {
            assert!(
                contains(got, (down, up)),
                "{} {name}: [{:.17e}, {:.17e}] does not contain [{:.17e}, {:.17e}]",
                case.label,
                got.inf(),
                got.sup(),
                down,
                up
            );
        }
    }
}

/// `c_0`, `c_1` and `2 c_2` against libcbe's energy, `g . v` and `v . H v`,
/// with the roundoff libcbe declares, `ops * eps * scale` per component, and
/// the rounding of the dot products.
#[test]
#[ignore = "libcbe's gradient and product convert with the truncated factor 0.0201723 \
            where its energy uses 0.03812 * 0.52918; waits on the potlibc wrapper fix \
            and a regenerated reference.txt"]
fn the_low_orders_match_the_compiled_energy_gradient_and_product() {
    let (_, cases) = reference();
    let eps = f64::EPSILON;
    for case in cases {
        let jet = Ch52008.taylor_jet(&case.x, &case.v, 2).expect("CH5 jet");
        let component = case.ops * eps * case.scale;
        let norm_v: f64 = case.v.iter().map(|v| v.abs()).sum();
        let n = case.v.len() as f64;
        let dot = |w: &[f64]| -> (f64, f64) {
            let value = w.iter().zip(&case.v).map(|(a, v)| a * v).sum();
            let terms: f64 = w.iter().zip(&case.v).map(|(a, v)| (a * v).abs()).sum();
            (value, component * norm_v + n * eps * terms)
        };
        let (slope, slope_bar) = dot(&case.libcbe_gradient);
        let (curvature, curvature_bar) = dot(&case.libcbe_hvp);
        let checks = [
            (
                "E",
                interval(&jet[0]),
                case.libcbe_energy,
                case.ops * eps * case.scale.max(case.libcbe_energy.abs()),
            ),
            ("g . v", interval(&jet[1]), slope, slope_bar),
            (
                "v . H v",
                point(2.0) * interval(&jet[2]),
                curvature,
                curvature_bar,
            ),
        ];
        for (name, got, want, bar) in checks {
            let miss = gap(got, want);
            eprintln!(
                "{} {name}: libcbe {want:.17e} gap {miss:.2e} bar {bar:.2e}",
                case.label
            );
            assert!(miss <= bar, "{} {name}", case.label);
        }
    }
}

unsafe extern "C" fn no_force(
    _: *mut std::os::raw::c_void,
    _: *const rgpot_force_input_t,
    _: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    rgpot_status_t::RGPOT_INTERNAL_ERROR
}

fn through_handle(
    mut pos: Vec<f64>,
    mut z: Vec<i32>,
    direction: &[f64],
    order: usize,
) -> (rgpot_status_t, Vec<f64>, Vec<f64>) {
    let mut cell = [0.0; 9];
    let pot = unsafe { rgpot_potential_new(no_force, std::ptr::null_mut(), None) };
    unsafe { rgpot_potential_set_taylor_jet(pot, Some(rgpot_ch5_2008_taylor_jet)) };
    let mut input = unsafe {
        rgpot_force_input_create(z.len(), pos.as_mut_ptr(), z.as_mut_ptr(), cell.as_mut_ptr())
    };
    let mut lower = vec![f64::NAN; order + 1];
    let mut upper = vec![f64::NAN; order + 1];
    let status = unsafe {
        rgpot_potential_taylor_jet(
            pot,
            &input,
            direction.as_ptr(),
            order,
            lower.as_mut_ptr(),
            upper.as_mut_ptr(),
        )
    };
    unsafe {
        rgpot_force_input_free(&mut input);
        rgpot_potential_free(pot);
    }
    (status, lower, upper)
}

#[test]
fn the_c_callback_serves_the_same_jet() {
    let v = mixed();
    let (status, lower, upper) =
        through_handle(SADDLE.to_vec(), CH5_2008_ATOMIC_NUMBERS.to_vec(), &v, 6);
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let direct = Ch52008.taylor_jet(&SADDLE, &v, 6).expect("jet");
    for (k, c) in direct.iter().enumerate() {
        assert_eq!((lower[k], upper[k]), (c.lo, c.hi), "c_{k}");
    }
}

#[test]
fn another_atom_order_is_refused() {
    let (status, ..) = through_handle(SADDLE.to_vec(), vec![6, 1, 1, 1, 1, 1], &mixed(), 2);
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    let message = unsafe { CStr::from_ptr(rgpot_last_error()) }
        .to_str()
        .unwrap();
    assert_eq!(
        message,
        "CH5 2008 atom order is [1, 6, 1, 1, 1, 1], got [6, 1, 1, 1, 1, 1]"
    );
    let (status, ..) = through_handle(SADDLE[..15].to_vec(), vec![1; 5], &mixed()[..15], 2);
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
}

#[test]
fn coincident_atoms_and_short_inputs_are_typed_refusals() {
    let mut x = SADDLE.to_vec();
    x.copy_within(3..6, 15);
    match Ch52008.taylor_jet(&x, &mixed(), 4) {
        Err(JetError::Domain { .. } | JetError::Branch { .. }) => {}
        other => panic!("coincident carbon and attacker: {other:?}"),
    }
    let (status, ..) = through_handle(x, CH5_2008_ATOMIC_NUMBERS.to_vec(), &mixed(), 4);
    assert!(
        matches!(
            status,
            rgpot_status_t::RGPOT_JET_DOMAIN | rgpot_status_t::RGPOT_JET_BRANCH
        ),
        "{status:?}"
    );
    assert_eq!(
        Ch52008.taylor_jet(&SADDLE[..9], &mixed()[..9], 4),
        Err(JetError::Shape("CH5 2008 takes 6 atoms"))
    );
}

/// A count of coefficients that fits in doubles but not in intervals panics
/// inside the jet. The two output addresses are never written: they are only
/// far enough apart to pass the overlap check.
#[cfg(target_pointer_width = "64")]
#[test]
fn a_panic_in_the_jet_is_an_internal_error_with_a_message() {
    let mut pos = SADDLE.to_vec();
    let mut z = CH5_2008_ATOMIC_NUMBERS.to_vec();
    let mut cell = [0.0; 9];
    let mut input = unsafe {
        rgpot_force_input_create(z.len(), pos.as_mut_ptr(), z.as_mut_ptr(), cell.as_mut_ptr())
    };
    let v = mixed();
    let count = 3usize << 58;
    let lower = 8 as *mut f64;
    let upper = (8 + count * std::mem::size_of::<f64>()) as *mut f64;
    unsafe { rgpot_core::status::rgpot_set_last_error(std::ptr::null()) };
    let status = unsafe {
        rgpot_ch5_2008_taylor_jet(
            std::ptr::null_mut(),
            &input,
            v.as_ptr(),
            count - 1,
            lower,
            upper,
        )
    };
    let message = unsafe { CStr::from_ptr(rgpot_last_error()) }
        .to_str()
        .unwrap()
        .to_owned();
    unsafe { rgpot_force_input_free(&mut input) };
    assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR, "{message:?}");
    assert!(message.contains("capacity overflow"), "{message:?}");
}
