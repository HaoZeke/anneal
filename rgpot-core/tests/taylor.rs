// MIT License
// Copyright 2023--present rgpot developers

//! Interval Taylor recurrences against exact series coefficients.
//!
//! Every reference is an exact rational enclosed by one outward-rounded
//! division, or a closed form evaluated in intervals independently of the
//! recurrences, so a recurrence passes only if its enclosure meets the
//! exact coefficient.

use rgpot_core::jet::JetError;
use rgpot_core::taylor::{side, Interval, Side, Taylor};

const ORDER: usize = 12;

fn rational(p: i64, q: i64) -> Interval {
    Interval::new(p as f64, p as f64) / Interval::new(q as f64, q as f64)
}

fn assert_series(got: &Taylor, want: &[Interval], label: &str) {
    for (k, (g, w)) in got.coefficients().iter().zip(want).enumerate() {
        eprintln!(
            "{label} t^{k}: [{:.17e}, {:.17e}] width {:.3e}",
            g.inf(),
            g.sup(),
            g.wid()
        );
        assert!(g.is_bounded(), "{label} t^{k} is unbounded");
        assert!(
            !g.disjoint(*w),
            "{label} t^{k} misses the exact coefficient"
        );
    }
}

fn odd_series(odd: &[(i64, i64)]) -> Vec<Interval> {
    let mut out = vec![Interval::ZERO; ORDER + 1];
    for (m, (p, q)) in odd.iter().enumerate() {
        if 2 * m < ORDER {
            out[2 * m + 1] = rational(*p, *q);
        }
    }
    out
}

#[test]
fn exp_matches_the_inverse_factorials() {
    let mut factorial = 1i64;
    let mut want = Vec::new();
    for k in 0..=ORDER {
        if k > 0 {
            factorial *= k as i64;
        }
        want.push(rational(1, factorial));
    }
    assert_series(&Taylor::line(0.0, 1.0, ORDER).exp(), &want, "exp(t)");
    let e = Interval::ONE.exp();
    let shifted: Vec<Interval> = want.iter().map(|w| e * *w).collect();
    assert_series(&Taylor::line(1.0, 1.0, ORDER).exp(), &shifted, "exp(1 + t)");
}

#[test]
fn sqrt_and_quotient_match_the_binomial_series() {
    let binomial = [
        (1, 1),
        (1, 2),
        (-1, 8),
        (1, 16),
        (-5, 128),
        (7, 256),
        (-21, 1024),
        (33, 2048),
        (-429, 32768),
        (715, 65536),
        (-2431, 262144),
        (4199, 524288),
        (-29393, 4194304),
    ];
    let want: Vec<Interval> = binomial.iter().map(|(p, q)| rational(*p, *q)).collect();
    let root = Taylor::line(1.0, 1.0, ORDER).sqrt().expect("sqrt(1 + t)");
    assert_series(&root, &want, "sqrt(1 + t)");
    let one = Taylor::constant(Interval::ONE, ORDER);
    let geometric = one
        .div(&Taylor::line(1.0, -1.0, ORDER))
        .expect("1 / (1 - t)");
    assert_series(&geometric, &vec![Interval::ONE; ORDER + 1], "1 / (1 - t)");
}

#[test]
fn tanh_and_acos_match_their_odd_series() {
    let tanh = odd_series(&[
        (1, 1),
        (-1, 3),
        (2, 15),
        (-17, 315),
        (62, 2835),
        (-1382, 155925),
    ]);
    assert_series(&Taylor::line(0.0, 1.0, ORDER).tanh(), &tanh, "tanh(t)");
    let mut acos = odd_series(&[
        (-1, 1),
        (-1, 6),
        (-3, 40),
        (-5, 112),
        (-35, 1152),
        (-63, 2816),
    ]);
    acos[0] = Interval::ZERO.acos();
    let got = Taylor::line(0.0, 1.0, ORDER).acos().expect("acos(t)");
    assert_series(&got, &acos, "acos(t)");
}

#[test]
fn polynomials_end_in_exact_zeros() {
    let x = Taylor::line(0.5, 2.0, ORDER);
    let quartic = x.powi(4);
    let want = [
        rational(1, 16),
        rational(1, 1),
        rational(6, 1),
        rational(16, 1),
        rational(16, 1),
    ];
    assert_series(&quartic, &want, "(1/2 + 2t)^4");
    for (k, c) in quartic.coefficients().iter().enumerate().skip(5) {
        assert_eq!(c.bounds(), (0.0, 0.0), "t^{k} of a quartic is not zero");
    }
}

#[test]
fn edges_of_the_analytic_domain_are_refused() {
    let at_zero = Taylor::line(0.0, 1.0, ORDER);
    match at_zero.sqrt() {
        Err(JetError::Domain {
            function: "sqrt", ..
        }) => {}
        other => panic!("sqrt at zero has no Taylor series: {other:?}"),
    }
    match Taylor::constant(Interval::ONE, ORDER).div(&at_zero) {
        Err(JetError::Domain {
            function: "division",
            ..
        }) => {}
        other => panic!("1/t has no Taylor series at zero: {other:?}"),
    }
    match Taylor::line(1.0, 1.0, ORDER).acos() {
        Err(JetError::Domain {
            function: "acos", ..
        }) => {}
        other => panic!("acos at one has no Taylor series: {other:?}"),
    }
    assert!(matches!(
        side(Interval::new(1.0, 2.0), 3.0, "x < 3"),
        Ok(Side::Below)
    ));
    assert!(matches!(
        side(Interval::new(4.0, 5.0), 3.0, "x < 3"),
        Ok(Side::Above)
    ));
    match side(Interval::new(2.0, 3.0), 3.0, "x < 3") {
        Err(JetError::Branch { test: "x < 3", .. }) => {}
        other => panic!("an enclosure touching the threshold is not decided: {other:?}"),
    }
}

/// `d^k/du^k (1 - T^2)` as integer polynomials in `T = tanh(u)`.
fn sech_squared_polynomials() -> Vec<Vec<i64>> {
    let mut out = vec![vec![1, 0, -1]];
    for _ in 0..ORDER {
        let last = out.last().expect("first polynomial");
        let mut next = vec![0i64; last.len() + 1];
        for (power, coefficient) in last.iter().enumerate().skip(1) {
            let term = power as i64 * coefficient;
            next[power - 1] += term;
            next[power + 1] -= term;
        }
        out.push(next);
    }
    out
}

/// `1 - tanh^2(t)` at the top against the series of `sech^2`, and at
/// `u = 0.3` against the derivative polynomials of `tanh` divided by `k!`.
#[test]
fn sech_squared_composes_tanh_and_products() {
    let sech2 = |u: f64| {
        let t = Taylor::line(u, 1.0, ORDER).tanh();
        Interval::ONE - &(&t * &t)
    };
    let top = [
        (1, 1),
        (0, 1),
        (-1, 1),
        (0, 1),
        (2, 3),
        (0, 1),
        (-17, 45),
        (0, 1),
        (62, 315),
        (0, 1),
        (-1382, 14175),
        (0, 1),
        (21844, 467775),
    ];
    let want: Vec<Interval> = top.iter().map(|(p, q)| rational(*p, *q)).collect();
    assert_series(&sech2(0.0), &want, "sech^2(t)");

    let u = 0.3;
    let t = Interval::new(u, u).tanh();
    let mut factorial = Interval::ONE;
    let mut want = Vec::new();
    for (k, polynomial) in sech_squared_polynomials().iter().enumerate() {
        if k > 0 {
            factorial = factorial * Interval::new(k as f64, k as f64);
        }
        let value = polynomial.iter().rev().fold(Interval::ZERO, |acc, c| {
            acc * t + Interval::new(*c as f64, *c as f64)
        });
        want.push(value / factorial);
    }
    assert_series(&sech2(u), &want, "sech^2(0.3 + t)");
}

/// The quartic double well `V0 ((x / x0)^2 - 1)^2` at its top and at its
/// minimum `x = x0`, with `V0 = 2` and `x0^2 = 50`.
#[test]
fn the_quartic_well_is_a_polynomial_in_t() {
    let v0 = Interval::new(2.0, 2.0);
    let x0 = Interval::new(50.0, 50.0).sqrt();
    let well = |x: Interval| {
        let mut line = Taylor::constant(x, ORDER);
        line = &line + &Taylor::line(0.0, 1.0, ORDER);
        let u = &line * (Interval::ONE / x0);
        let w = &(&u * &u) - Interval::ONE;
        &(&w * &w) * v0
    };
    let mut top = vec![Interval::ZERO; ORDER + 1];
    top[0] = rational(2, 1);
    top[2] = rational(-2, 25);
    top[4] = rational(1, 1250);
    let at_top = well(Interval::ZERO);
    assert_series(&at_top, &top, "double well top");
    for k in [1, 3, 5, 6, 7, 8, 9, 10, 11, 12] {
        assert_eq!(at_top.coefficients()[k].bounds(), (0.0, 0.0), "top t^{k}");
    }
    let mut bottom = vec![Interval::ZERO; ORDER + 1];
    bottom[2] = rational(8, 50);
    bottom[3] = Interval::new(8.0, 8.0) / (Interval::new(50.0, 50.0) * x0);
    bottom[4] = rational(2, 2500);
    assert_series(&well(x0), &bottom, "double well minimum");
}
