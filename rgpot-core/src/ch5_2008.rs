// MIT License
// Copyright 2023--present rgpot developers

//! Exact Taylor jet of the CH5 2008 surface.
//!
//! This is `POT` of `ch52008.f`, the CH4 + H surface of Corchado, Bravo and
//! Espinosa-Garcia, J. Chem. Phys. 130, 184314 (2009), read in exact real
//! arithmetic with every literal taken as the value gfortran assigns it.
//! `ANUZERO` is a default-real literal, so it is the binary32 value. The
//! source's hand-written gradients are not transcribed: the series carries
//! every derivative of the energy. Positions are Bohr and the energy is
//! Hartree, as in the source.
//!
//! The source compares in four places: the `tanh` saturation guards
//! (`arg < 19`), the `rch < 3.8` switch, and the right-handedness swap of
//! the out-of-plane bend (`argd > 0`). Each is decided on the enclosure of
//! its argument; an enclosure that contains the threshold is
//! [`JetError::Branch`].

use std::os::raw::c_void;

use crate::jet::{rgpot_interval_t, serve_taylor_jet, JetError, TaylorJet};
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::taylor::{side, Interval, Side, Taylor};
use crate::tensor::copy_i32_tensor;
use crate::types::rgpot_force_input_t;

type Vector = [Taylor; 3];

fn lit(value: f64) -> Interval {
    Interval::new(value, value)
}

/// Carbon, the attacking hydrogen and the four methane hydrogens, as the
/// zero-based atom indices of the source's `BLOCK DATA` (`nnc = 2`,
/// `nnb = 6`, `nnh = 3, 4, 5, 1`).
const CARBON: usize = 1;
const ATTACKER: usize = 5;
const METHANE: [usize; 4] = [2, 3, 4, 0];

struct Parameters {
    r0ch: Interval,
    d1ch: Interval,
    d3ch: Interval,
    a1ch: Interval,
    b1ch: Interval,
    c1ch: Interval,
    r0hh: Interval,
    d1hh: Interval,
    d3hh: Interval,
    ahh: Interval,
    r0cb: Interval,
    d1cb: Interval,
    d3cb: Interval,
    acb: Interval,
    a3s: Interval,
    b3s: Interval,
    aphi: Interval,
    bphi: Interval,
    cphi: Interval,
    atheta: Interval,
    btheta: Interval,
    ctheta: Interval,
    fch3: Interval,
    hch3: Interval,
    fkinf: Interval,
    ak: Interval,
    aa1: Interval,
    aa2: Interval,
    aa3: Interval,
    aa4: Interval,
    a1s: Interval,
    b1s: Interval,
    a2s: Interval,
    b2s: Interval,
}

impl Parameters {
    /// `BLOCK DATA PTPACM`, scaled as `PREPOT` scales it, and `switchf`'s constants.
    fn new() -> Self {
        let fact1 = lit(0.041840);
        let fact2 = lit(6.022045);
        Parameters {
            r0ch: lit(1.08898),
            d1ch: lit(111.266) * fact1,
            d3ch: lit(48.96226) * fact1,
            a1ch: lit(1.78374),
            b1ch: lit(0.14201),
            c1ch: lit(2.21773),
            r0hh: lit(0.74239),
            d1hh: lit(108.382) * fact1,
            d3hh: lit(38.42657) * fact1,
            ahh: lit(1.9589),
            r0cb: lit(1.08898),
            d1cb: lit(56.505) * fact1,
            d3cb: lit(19.612) * fact1,
            acb: lit(1.4173200),
            a3s: lit(0.1475300),
            b3s: lit(-2.9926300),
            aphi: lit(0.5307000),
            bphi: lit(0.4012200),
            cphi: lit(1.9235100),
            atheta: lit(0.9119800),
            btheta: lit(0.3537500),
            ctheta: lit(1.8970500),
            fch3: lit(0.0693700) * fact2,
            hch3: lit(0.1387400) * fact2,
            fkinf: lit(0.4291400) * fact2,
            ak: lit(0.1353000) * fact2,
            aa1: lit(1.265960),
            aa2: lit(0.000710),
            aa3: lit(0.985920),
            aa4: lit(2.785060),
            a1s: lit(1.5132681e-7),
            b1s: lit(-4.3792246),
            a2s: lit(1.9202402e-7),
            b2s: lit(-12.323018),
        }
    }
}

fn difference(a: &Vector, b: &Vector) -> Vector {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}

fn dot(a: &Vector, b: &Vector) -> Taylor {
    &(&(&a[0] * &b[0]) + &(&a[1] * &b[1])) + &(&a[2] * &b[2])
}

fn cross(a: &Vector, b: &Vector) -> Vector {
    [
        &(&a[1] * &b[2]) - &(&a[2] * &b[1]),
        &(&a[2] * &b[0]) - &(&a[0] * &b[2]),
        &(&a[0] * &b[1]) - &(&a[1] * &b[0]),
    ]
}

fn length(a: &Vector) -> Result<Taylor, JetError> {
    dot(a, a).sqrt()
}

/// `1 - tanh(arg)` below the saturation guard `arg < 19`, zero above it.
fn saturated(arg: &Taylor, test: &'static str) -> Result<Taylor, JetError> {
    match side(arg.value(), 19.0, test)? {
        Side::Below => Ok(Interval::ONE - &arg.tanh()),
        Side::Above => Ok(Taylor::constant(Interval::ZERO, arg.order())),
    }
}

/// The two indices of `0..4` other than `i` and `j`.
fn complement(i: usize, j: usize) -> (usize, usize) {
    let mut rest = (0..4).filter(|m| *m != i && *m != j);
    match (rest.next(), rest.next()) {
        (Some(k), Some(l)) => (k, l),
        _ => unreachable!("two of four indices remain"),
    }
}

/// Index of the unordered pair `{i, j}`, `i != j`, in `0..6`.
fn pair(i: usize, j: usize) -> usize {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    match (a, b) {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 3) => 2,
        (1, 2) => 3,
        (1, 3) => 4,
        (2, 3) => 5,
        _ => unreachable!("a pair of distinct indices below four"),
    }
}

/// `(vq, vj)` of one LEPS bond: singlet `d1`, triplet `d3`, range `a`.
fn leps(d1: Interval, d3: Interval, a: &Taylor, r: &Taylor, r0: Interval) -> (Taylor, Taylor) {
    let stretch = r - r0;
    let double = (&(a * &stretch) * lit(-2.0)).exp();
    let single = (-&(a * &stretch)).exp();
    let e1 = &(&double - &(&single * lit(2.0))) * d1;
    let e3 = &(&double + &(&single * lit(2.0))) * d3;
    (&(&e1 + &e3) * lit(0.5), &(&e1 - &e3) * lit(0.5))
}

/// Energy coefficients of `V(x + t v)`, `x` and `v` in Bohr, through `order`.
fn line_jet(x: &[f64], v: &[f64], order: usize) -> Result<Taylor, JetError> {
    if x.len() != 18 || v.len() != 18 {
        return Err(JetError::Shape("CH5 2008 takes 6 atoms"));
    }
    let p = Parameters::new();
    let angstrom = lit(0.52918);
    let position = |atom: usize| -> Vector {
        let at = |c: usize| &Taylor::line(x[3 * atom + c], v[3 * atom + c], order) * angstrom;
        [at(0), at(1), at(2)]
    };
    let carbon = position(CARBON);
    let attacker = position(ATTACKER);
    let methane: Vec<Vector> = METHANE.iter().map(|atom| position(*atom)).collect();
    let constant = |value: Interval| Taylor::constant(value, order);

    // coorden
    let tcb = difference(&carbon, &attacker);
    let tch: Vec<Vector> = methane.iter().map(|h| difference(&carbon, h)).collect();
    let tbh: Vec<Vector> = methane.iter().map(|h| difference(&attacker, h)).collect();
    let rcb = length(&tcb)?;
    let rch = tch.iter().map(length).collect::<Result<Vec<_>, _>>()?;
    let rbh = tbh.iter().map(length).collect::<Result<Vec<_>, _>>()?;

    // switchf
    let mut s1 = Vec::with_capacity(4);
    let mut s2 = Vec::with_capacity(4);
    let mut s3 = Vec::with_capacity(4);
    let mut sphi = Vec::with_capacity(4);
    let mut stheta = Vec::with_capacity(4);
    for r in &rch {
        let dr = r - p.r0ch;
        s1.push(saturated(
            &(&(&dr * p.a1s) * &(r - p.b1s).powi(8)),
            "args1 < 19",
        )?);
        s2.push(saturated(
            &(&(&dr * p.a2s) * &(r - p.b2s).powi(6)),
            "args2 < 19",
        )?);
        s3.push(saturated(
            &(&(&dr * p.a3s) * &(r - p.b3s).powi(2)),
            "args3 < 19",
        )?);
        match side(r.value(), 3.8, "rch < 3.8")? {
            Side::Below => {
                let phi = &(&dr * p.aphi) * &(&(r - p.cphi).powi(3) * p.bphi).exp();
                let theta = &(&dr * p.atheta) * &(&(r - p.ctheta).powi(3) * p.btheta).exp();
                sphi.push(Interval::ONE - &phi.tanh());
                stheta.push(Interval::ONE - &theta.tanh());
            }
            Side::Above => {
                sphi.push(constant(Interval::ZERO));
                stheta.push(constant(Interval::ZERO));
            }
        }
    }

    // refangles
    // The source's 3.141592653589793D0 is this binary64 value.
    let pi = lit(std::f64::consts::PI);
    let tau = (lit(-1.0) / lit(3.0)).acos();
    let toward_planar = tau - lit(0.5) * pi;
    let toward_trigonal = tau - lit(2.0) * pi / lit(3.0);
    let mut theta0 = Vec::with_capacity(6);
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let (k, l) = complement(i, j);
        let planar = &(&(&sphi[i] * &sphi[j]) - Interval::ONE) * toward_planar;
        let trigonal = &(&(&stheta[k] * &stheta[l]) - Interval::ONE) * toward_trigonal;
        theta0.push(&(&planar + &trigonal) + tau);
    }

    // stretch
    let rav = &(&(&(&rch[0] + &rch[1]) + &rch[2]) + &rch[3]) * lit(0.25);
    let arga = &(&rav - p.r0ch) * p.c1ch;
    let ach = match side(arga.value(), 19.0, "arga < 19")? {
        Side::Below => &(&(&arga.tanh() + Interval::ONE) * (p.b1ch * lit(0.5))) + p.a1ch,
        Side::Above => constant(p.a1ch + p.b1ch),
    };
    let (vqcb, vjcb) = leps(p.d1cb, p.d3cb, &constant(p.acb), &rcb, p.r0cb);
    let mut vstr = constant(Interval::ZERO);
    for i in 0..4 {
        let (vqch, vjch) = leps(p.d1ch, p.d3ch, &ach, &rch[i], p.r0ch);
        let (vqbh, vjbh) = leps(p.d1hh, p.d3hh, &constant(p.ahh), &rbh[i], p.r0hh);
        let vq = &(&vqch + &vqcb) + &vqbh;
        let spread = &(&(&(&vjch - &vjcb).powi(2) + &(&vjcb - &vjbh).powi(2))
            + &(&vjbh - &vjch).powi(2))
            * lit(0.5);
        let vj = -&spread.sqrt()?;
        vstr = &(&vstr + &vq) + &vj;
    }

    // opforce and opbend
    let mut vop = constant(Interval::ZERO);
    let unit: Vec<Vector> = tch
        .iter()
        .zip(&rch)
        .map(|(t, r)| Ok([(-&t[0]).div(r)?, (-&t[1]).div(r)?, (-&t[2]).div(r)?]))
        .collect::<Result<Vec<_>, JetError>>()?;
    for i in 0..4 {
        let mut others = constant(Interval::ONE);
        for (m, s) in s3.iter().enumerate() {
            if m != i {
                others = &others * s;
            }
        }
        let switch = &(Interval::ONE - &s3[i]) * &others;
        let (j, mut k, mut l) = ((i + 1) % 4, (i + 2) % 4, (i + 3) % 4);
        let normal = cross(
            &difference(&methane[k], &methane[j]),
            &difference(&methane[l], &methane[j]),
        );
        let norma = length(&normal)?;
        for m in [j, k, l] {
            let argd = dot(&normal, &unit[m]).div(&norma)?;
            if side(argd.value(), 0.0, "argd > 0")? == Side::Above {
                std::mem::swap(&mut k, &mut l);
            }
        }
        let normal = cross(
            &difference(&methane[k], &methane[j]),
            &difference(&methane[l], &methane[j]),
        );
        let norma = length(&normal)?;
        let mut sum2 = constant(Interval::ZERO);
        let mut sum4 = constant(Interval::ZERO);
        for m in [j, k, l] {
            let argd = dot(&normal, &unit[m]).div(&norma)?;
            let delta = &argd.acos()? - &theta0[pair(i, m)];
            let square = &delta * &delta;
            sum4 = &sum4 + &(&square * &square);
            sum2 = &sum2 + &square;
        }
        vop = &vop + &(&(&(&sum2 * p.fch3) + &(&sum4 * p.hch3)) * &switch);
    }

    // ipforce and ipbend
    let f0 = p.fkinf + p.ak;
    let f2 = p.fkinf;
    let mut f1 = Vec::with_capacity(4);
    for (r, b) in rch.iter().zip(&rbh) {
        let a1 = Interval::ONE - &(&(b * b) * (-p.aa1)).exp();
        let a2 = &(&(&(b - p.r0hh).powi(2) * (-p.aa4)).exp() * p.aa3) + p.aa2;
        let stretch = (r - p.r0ch).powi(2);
        f1.push(&a1 * &(-&(&a2 * &stretch)).exp());
    }
    let mut vip = constant(Interval::ZERO);
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let (k, l) = complement(i, j);
        let fk0 = &(&(&(&(&s1[i] * &s1[j]) - Interval::ONE) * f0)
            + &(&(&(&s2[k] * &s2[l]) - Interval::ONE) * (f0 - f2)))
            + f0;
        let costh = dot(&tch[i], &tch[j]).div(&(&rch[i] * &rch[j]))?;
        let dth = &costh.acos()? - &theta0[pair(i, j)];
        let term = &(&(&fk0 * &f1[i]) * &f1[j]) * &(&dth * &dth);
        vip = &vip + &(&term * lit(0.5));
    }

    // The binary32 rounding of the source's -0.70983241091685123.
    let anuzero = lit(f64::from(-0.709_832_43_f32));
    Ok(&(&(&(&vstr + &vop) + &vip) * lit(0.03812)) - anuzero)
}

/// The CH5 2008 surface, Bohr in and Hartree out.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ch52008;

impl TaylorJet for Ch52008 {
    fn taylor_jet(
        &self,
        positions: &[f64],
        direction: &[f64],
        order: usize,
    ) -> Result<Vec<rgpot_interval_t>, JetError> {
        Ok(line_jet(positions, direction, order)?.into_enclosures())
    }
}

/// The atomic numbers of the source's atom order.
pub const CH5_2008_ATOMIC_NUMBERS: [i32; 6] = [1, 6, 1, 1, 1, 1];

/// Taylor jet callback of the CH5 2008 surface, for
/// `rgpot_potential_set_taylor_jet` on a handle that evaluates it.
///
/// Positions are six atoms in Bohr in the order of the source's
/// `BLOCK DATA`: the fourth methane hydrogen, carbon, the first three
/// methane hydrogens and the attacking hydrogen. The coefficients are
/// in Hartree. Atomic numbers, when given, must be 1, 6, 1, 1, 1, 1; the
/// cell is not read and `user_data` is ignored.
///
/// # Safety
///
/// As for `rgpot_taylor_jet_fn`: NULL pointers and overlapping `lower` and
/// `upper` return `RGPOT_INVALID_PARAMETER`. Otherwise `input` must point to
/// a valid `rgpot_force_input_t`, `direction` to 18 readable doubles, and
/// `lower` and `upper` to `order + 1` writable doubles each.
#[no_mangle]
pub unsafe extern "C" fn rgpot_ch5_2008_taylor_jet(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    let _ = user_data;
    catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        serve_ch5_2008(input, direction, order, lower, upper)
    }))
}

unsafe fn serve_ch5_2008(
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    if !input.is_null() {
        let input = unsafe { &*input };
        if unsafe { input.n_atoms() } != Some(6) {
            set_last_error("CH5 2008 takes 6 atoms");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if !input.atomic_numbers.is_null() {
            match copy_i32_tensor(input.atomic_numbers, "atomic_numbers", 6) {
                Ok(z) if z == CH5_2008_ATOMIC_NUMBERS => {}
                Ok(z) => {
                    set_last_error(&format!(
                        "CH5 2008 atom order is {CH5_2008_ATOMIC_NUMBERS:?}, got {z:?}"
                    ));
                    return rgpot_status_t::RGPOT_INVALID_PARAMETER;
                }
                Err(why) => {
                    set_last_error(&why);
                    return rgpot_status_t::RGPOT_INVALID_PARAMETER;
                }
            }
        }
    }
    unsafe { serve_taylor_jet(&Ch52008, input, direction, order, lower, upper) }
}
