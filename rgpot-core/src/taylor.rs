// MIT License
// Copyright 2023--present rgpot developers

//! Truncated Taylor series with interval coefficients.
//!
//! A [`Taylor`] holds enclosures of the coefficients `u_0, ..., u_K` of a
//! function of one variable `t` at `t = 0`. Sums, products, quotients,
//! `sqrt`, `exp`, `tanh` and `acos` propagate them by the recurrences of
//! Griewank and Walther, *Evaluating Derivatives*, 2nd ed., 2008, section
//! 13.2. Every operation is outward rounded by `maryada`, so each
//! coefficient of a result encloses the exact coefficient of the composed
//! function. There is no remainder: the coefficients through `K` of a
//! composition depend only on the coefficients through `K` of its arguments.
//!
//! A quotient, a square root or an arc cosine whose argument enclosure
//! touches the edge of the function's analytic domain is refused with
//! [`JetError::Domain`], and a comparison whose argument enclosure contains
//! the threshold with [`JetError::Branch`]: near such a point the function
//! is not one analytic expression. A potential evaluates its energy on
//! [`Taylor::line`] coordinates to implement [`crate::jet::TaylorJet`].
//!
//! Series combined by one operation must have the same order; mixing orders
//! is a programming error and panics.

use std::ops::{Add, Mul, Neg, Sub};

pub use maryada::Interval;

use crate::jet::{rgpot_interval_t, JetError};

/// Side of a threshold on which an enclosure lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Below,
    Above,
}

/// The side of `threshold` on which all of `value` lies.
pub fn side(value: Interval, threshold: f64, test: &'static str) -> Result<Side, JetError> {
    if value.is_empty() {
        Err(JetError::Branch {
            test,
            lo: value.inf(),
            hi: value.sup(),
        })
    } else if value.sup() < threshold {
        Ok(Side::Below)
    } else if value.inf() > threshold {
        Ok(Side::Above)
    } else {
        Err(JetError::Branch {
            test,
            lo: value.inf(),
            hi: value.sup(),
        })
    }
}

fn domain(function: &'static str, value: Interval) -> JetError {
    JetError::Domain {
        function,
        lo: value.inf(),
        hi: value.sup(),
    }
}

fn point(value: f64) -> Interval {
    Interval::new(value, value)
}

/// Enclosures of `u_0, ..., u_K`.
#[derive(Debug, Clone)]
pub struct Taylor {
    c: Vec<Interval>,
}

impl Taylor {
    /// The constant `value` through order `order`.
    pub fn constant(value: Interval, order: usize) -> Self {
        let mut c = vec![Interval::ZERO; order + 1];
        c[0] = value;
        Taylor { c }
    }

    /// `x + t v` through order `order`.
    pub fn line(x: f64, v: f64, order: usize) -> Self {
        let mut c = vec![Interval::ZERO; order + 1];
        c[0] = point(x);
        if order > 0 {
            c[1] = point(v);
        }
        Taylor { c }
    }

    /// The highest order carried.
    pub fn order(&self) -> usize {
        self.c.len() - 1
    }

    /// The enclosure of the value at `t = 0`.
    pub fn value(&self) -> Interval {
        self.c[0]
    }

    /// The coefficient enclosures `u_0, ..., u_K`.
    pub fn coefficients(&self) -> &[Interval] {
        &self.c
    }

    /// The coefficient enclosures, by value.
    pub fn into_coefficients(self) -> Vec<Interval> {
        self.c
    }

    /// The coefficients as the bounds a jet callback writes.
    pub fn into_enclosures(self) -> Vec<rgpot_interval_t> {
        self.c
            .iter()
            .map(|c| rgpot_interval_t {
                lo: c.inf(),
                hi: c.sup(),
            })
            .collect()
    }

    fn same_order(&self, other: &Taylor) {
        assert_eq!(self.c.len(), other.c.len(), "Taylor orders differ");
    }

    /// `self^power`.
    pub fn powi(&self, power: u32) -> Taylor {
        let mut out = Taylor::constant(Interval::ONE, self.order());
        let mut base = self.clone();
        let mut rest = power;
        while rest > 0 {
            if rest & 1 == 1 {
                out = &out * &base;
            }
            rest >>= 1;
            if rest > 0 {
                base = &base * &base;
            }
        }
        out
    }

    /// `self / other`; refused when the divisor's value encloses zero.
    pub fn div(&self, other: &Taylor) -> Result<Taylor, JetError> {
        self.same_order(other);
        let v0 = other.c[0];
        if v0.is_empty() || v0.contains(0.0) {
            return Err(domain("division", v0));
        }
        let mut w = vec![Interval::ZERO; self.c.len()];
        for k in 0..w.len() {
            let mut sum = self.c[k];
            for j in 1..=k {
                sum = sum - other.c[j] * w[k - j];
            }
            w[k] = sum / v0;
        }
        Ok(Taylor { c: w })
    }

    /// `sqrt(self)`; refused unless the value's enclosure is positive.
    pub fn sqrt(&self) -> Result<Taylor, JetError> {
        let u0 = self.c[0];
        if u0.is_empty() || u0.inf() <= 0.0 {
            return Err(domain("sqrt", u0));
        }
        let mut w = vec![Interval::ZERO; self.c.len()];
        w[0] = u0.sqrt();
        let twice = point(2.0) * w[0];
        for k in 1..w.len() {
            let mut sum = self.c[k];
            for j in 1..k {
                sum = sum - w[j] * w[k - j];
            }
            w[k] = sum / twice;
        }
        Ok(Taylor { c: w })
    }

    /// `exp(self)` from `w' = w u'`.
    pub fn exp(&self) -> Taylor {
        let mut w = vec![Interval::ZERO; self.c.len()];
        w[0] = self.c[0].exp();
        for k in 1..w.len() {
            let mut sum = Interval::ZERO;
            for j in 1..=k {
                sum = sum + point(j as f64) * self.c[j] * w[k - j];
            }
            w[k] = sum / point(k as f64);
        }
        Taylor { c: w }
    }

    /// `tanh(self)` from `w' = (1 - w^2) u'`.
    pub fn tanh(&self) -> Taylor {
        let n = self.c.len();
        let mut w = vec![Interval::ZERO; n];
        let mut s = vec![Interval::ZERO; n];
        w[0] = self.c[0].tanh();
        s[0] = Interval::ONE - w[0].sqr();
        for k in 1..n {
            let mut sum = Interval::ZERO;
            for j in 1..=k {
                sum = sum + point(j as f64) * self.c[j] * s[k - j];
            }
            w[k] = sum / point(k as f64);
            let mut square = Interval::ZERO;
            for i in 0..=k {
                square = square + w[i] * w[k - i];
            }
            s[k] = -square;
        }
        Taylor { c: w }
    }

    /// `acos(self)` from `sqrt(1 - u^2) w' = -u'`; refused unless `|u_0| < 1`.
    pub fn acos(&self) -> Result<Taylor, JetError> {
        let one = Taylor::constant(Interval::ONE, self.order());
        let root = (&one - &(self * self))
            .sqrt()
            .map_err(|_| domain("acos", self.c[0]))?;
        let n = self.c.len();
        let mut w = vec![Interval::ZERO; n];
        w[0] = self.c[0].acos();
        for k in 1..n {
            let kk = point(k as f64);
            let mut sum = -(kk * self.c[k]);
            for (j, wj) in w.iter().enumerate().take(k).skip(1) {
                sum = sum - point(j as f64) * *wj * root.c[k - j];
            }
            w[k] = sum / (kk * root.c[0]);
        }
        Ok(Taylor { c: w })
    }
}

impl Add<&Taylor> for &Taylor {
    type Output = Taylor;
    fn add(self, other: &Taylor) -> Taylor {
        self.same_order(other);
        Taylor {
            c: self.c.iter().zip(&other.c).map(|(a, b)| *a + *b).collect(),
        }
    }
}

impl Sub<&Taylor> for &Taylor {
    type Output = Taylor;
    fn sub(self, other: &Taylor) -> Taylor {
        self.same_order(other);
        Taylor {
            c: self.c.iter().zip(&other.c).map(|(a, b)| *a - *b).collect(),
        }
    }
}

impl Mul<&Taylor> for &Taylor {
    type Output = Taylor;
    fn mul(self, other: &Taylor) -> Taylor {
        self.same_order(other);
        let n = self.c.len();
        let mut w = vec![Interval::ZERO; n];
        for (k, slot) in w.iter_mut().enumerate() {
            let mut sum = Interval::ZERO;
            for j in 0..=k {
                sum = sum + self.c[j] * other.c[k - j];
            }
            *slot = sum;
        }
        Taylor { c: w }
    }
}

impl Neg for &Taylor {
    type Output = Taylor;
    fn neg(self) -> Taylor {
        Taylor {
            c: self.c.iter().map(|a| -*a).collect(),
        }
    }
}

impl Add<Interval> for &Taylor {
    type Output = Taylor;
    fn add(self, other: Interval) -> Taylor {
        let mut c = self.c.clone();
        c[0] = c[0] + other;
        Taylor { c }
    }
}

impl Sub<Interval> for &Taylor {
    type Output = Taylor;
    fn sub(self, other: Interval) -> Taylor {
        let mut c = self.c.clone();
        c[0] = c[0] - other;
        Taylor { c }
    }
}

impl Mul<Interval> for &Taylor {
    type Output = Taylor;
    fn mul(self, other: Interval) -> Taylor {
        Taylor {
            c: self.c.iter().map(|a| *a * other).collect(),
        }
    }
}

macro_rules! owned {
    ($trait:ident, $method:ident, $rhs:ty) => {
        impl $trait<$rhs> for Taylor {
            type Output = Taylor;
            fn $method(self, other: $rhs) -> Taylor {
                (&self).$method(other)
            }
        }
    };
}

owned!(Add, add, &Taylor);
owned!(Sub, sub, &Taylor);
owned!(Mul, mul, &Taylor);
owned!(Add, add, Interval);
owned!(Sub, sub, Interval);
owned!(Mul, mul, Interval);

impl Add<Taylor> for Taylor {
    type Output = Taylor;
    fn add(self, other: Taylor) -> Taylor {
        &self + &other
    }
}

impl Sub<Taylor> for Taylor {
    type Output = Taylor;
    fn sub(self, other: Taylor) -> Taylor {
        &self - &other
    }
}

impl Mul<Taylor> for Taylor {
    type Output = Taylor;
    fn mul(self, other: Taylor) -> Taylor {
        &self * &other
    }
}

impl Neg for Taylor {
    type Output = Taylor;
    fn neg(self) -> Taylor {
        -&self
    }
}

impl Mul<&Taylor> for Interval {
    type Output = Taylor;
    fn mul(self, other: &Taylor) -> Taylor {
        other * self
    }
}

impl Mul<Taylor> for Interval {
    type Output = Taylor;
    fn mul(self, other: Taylor) -> Taylor {
        &other * self
    }
}

impl Sub<&Taylor> for Interval {
    type Output = Taylor;
    fn sub(self, other: &Taylor) -> Taylor {
        let mut c: Vec<Interval> = other.c.iter().map(|a| -*a).collect();
        c[0] = self - other.c[0];
        Taylor { c }
    }
}

impl Sub<Taylor> for Interval {
    type Output = Taylor;
    fn sub(self, other: Taylor) -> Taylor {
        self - &other
    }
}
