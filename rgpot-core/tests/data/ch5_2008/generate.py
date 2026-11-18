#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Reference data for rgpot-core/tests/ch5_2008.rs.

Writes, for deterministic geometries x and directions v near the H + CH4
saddle of the CH5 2008 surface:

- the Taylor coefficients c_k of V(x + t v), k = 0..ORDER, as the binary64
  bracket [down, up] of each exact value;
- the energy E, the slope g . v and the curvature v . H v, as binary64
  brackets of values taken from partial derivatives of the scalar energy;
- the energy, the analytic gradient and the curvature product H v of the
  compiled source (libcbe), with the operation count and scale that libcbe's
  product reports for its roundoff.

V is POT of ch52008.f (i-PI 3.3.1, drivers/f90/pes) with every literal read
as gfortran reads it and every operation exact: double precision literals are
their binary64 values, the default-real ANUZERO is its binary32 value, and
the PREPOT unit products, 2*pi, acos(-1/3) and the Bohr-to-Angstrom scaling
are exact real operations. It is transcribed here from the Fortran, not from
rgpot's Rust, and evaluated on truncated power series of mpmath numbers at 60
digits. Every coefficient is also computed by mpmath.taylor, which
differentiates the scalar function numerically at raised precision, and E,
g . v and v . H v by mpmath.diff on the 18 coordinates; c_0, c_1 and 2 c_2
must agree with those before anything is written.

libcbe's gradient and product convert with the truncated factor 0.0201723
where its energy uses 0.03812 * 0.52918 = 0.0201723416. Regenerate this file
from the fixed library when the potlibc wrapper fix lands.

Usage:

    python3 generate.py --libcbe build/libcbe.so \
        --source $IPI/drivers/f90/pes/ch52008.f --output reference.txt

libcbe is the batch library spectral-instanton builds from that source
(cbe_eval: energy and the NDER=1 gradient; cbe_hvp: the forward-mode dual of
that gradient). Requires mpmath 1.4.
"""

import argparse
import ctypes
import hashlib
import random
import struct
import sys

import mpmath
from mpmath import mp, mpf

ORDER = 12
DPS = 60
CHECK_DPS = 40

SADDLE = [
    2.5078986937601746, 3.0439878740728727, 2.603957609783651,
    1.1100980488026544, 1.3613545004038685, 1.1486133528262976,
    0.013880929393421192, 0.2086334261632169, 2.457516169583444,
    2.4611712350631576, 0.23586419033998002, 0.07550700261319757,
    -0.09665148160819977, 2.4937031206965017, -0.0782634154328256,
    3.3853544918540113, 4.100244654765713, 3.5175358525837166,
]


class Series:
    """Truncated power series in t with coefficients 0..ORDER."""

    def __init__(self, c):
        self.c = list(c)

    @staticmethod
    def const(a):
        return Series([mpf(a)] + [mpf(0)] * ORDER)

    @staticmethod
    def lift(a):
        return a if isinstance(a, Series) else Series.const(a)

    def __add__(self, o):
        o = Series.lift(o)
        return Series([a + b for a, b in zip(self.c, o.c)])

    __radd__ = __add__

    def __neg__(self):
        return Series([-a for a in self.c])

    def __sub__(self, o):
        return self + (-Series.lift(o))

    def __rsub__(self, o):
        return Series.lift(o) - self

    def __mul__(self, o):
        if not isinstance(o, Series):
            o = mpf(o)
            return Series([a * o for a in self.c])
        return Series(
            [sum(self.c[j] * o.c[k - j] for j in range(k + 1)) for k in range(ORDER + 1)]
        )

    __rmul__ = __mul__

    def __truediv__(self, o):
        if not isinstance(o, Series):
            o = mpf(o)
            return Series([a / o for a in self.c])
        q = []
        for k in range(ORDER + 1):
            q.append((self.c[k] - sum(q[j] * o.c[k - j] for j in range(k))) / o.c[0])
        return Series(q)

    def __rtruediv__(self, o):
        return Series.lift(o) / self

    def __pow__(self, n):
        out = Series.const(1)
        for _ in range(n):
            out = out * self
        return out

    def value(self):
        return self.c[0]


def derivative(a):
    return Series([(k + 1) * a.c[k + 1] for k in range(ORDER)] + [mpf(0)])


def integral(a, start):
    return Series([start] + [a.c[k - 1] / k for k in range(1, ORDER + 1)])


def s_exp(a):
    if not isinstance(a, Series):
        return mpmath.exp(a)
    e = [mpmath.exp(a.c[0])]
    for k in range(1, ORDER + 1):
        e.append(sum(j * a.c[j] * e[k - j] for j in range(1, k + 1)) / k)
    return Series(e)


def s_sqrt(a):
    if not isinstance(a, Series):
        return mpmath.sqrt(a)
    s = [mpmath.sqrt(a.c[0])]
    for k in range(1, ORDER + 1):
        s.append((a.c[k] - sum(s[j] * s[k - j] for j in range(1, k))) / (2 * s[0]))
    return Series(s)


def s_tanh(a):
    if not isinstance(a, Series):
        return mpmath.tanh(a)
    return 1 - 2 / (s_exp(2 * a) + 1)


def s_acos(a):
    if not isinstance(a, Series):
        return mpmath.acos(a)
    return integral(-derivative(a) / s_sqrt(1 - a * a), mpmath.acos(a.c[0]))


def lit(text):
    return mpf(float(text))


def default_real(text):
    single = mpmath.mpf(mpmath.libmp.from_str(text, 24, "n"))
    assert single == mpf(struct.unpack("f", struct.pack("f", float(text)))[0])
    return single


class Margins:
    """The smallest distance of each branch test from its threshold."""

    def __init__(self):
        self.saturation = mpmath.inf
        self.switch = mpmath.inf
        self.handedness = mpmath.inf

    def below(self, value, threshold, kind):
        gap = threshold - value
        setattr(self, kind, min(getattr(self, kind), abs(gap)))
        return gap > 0

    def above(self, value, threshold, kind):
        gap = value - threshold
        setattr(self, kind, min(getattr(self, kind), abs(gap)))
        return gap > 0


def energy(x, ops, margins):
    """POT of ch52008.f at Bohr positions x (18 entries), in Hartree."""
    sqrt, exp, tanh, acos = ops
    fact1 = lit("0.041840")
    fact2 = lit("6.022045")
    r0ch, d1ch, d3ch = lit("1.08898"), lit("111.266") * fact1, lit("48.96226") * fact1
    a1ch, b1ch, c1ch = lit("1.78374"), lit("0.14201"), lit("2.21773")
    r0hh, d1hh, d3hh = lit("0.74239"), lit("108.382") * fact1, lit("38.42657") * fact1
    ahh = lit("1.9589")
    r0cb, d1cb, d3cb = lit("1.08898"), lit("56.505") * fact1, lit("19.612") * fact1
    acb = lit("1.4173200")
    a3s, b3s = lit("0.1475300"), lit("-2.9926300")
    aphi, bphi, cphi = lit("0.5307000"), lit("0.4012200"), lit("1.9235100")
    atheta, btheta, ctheta = lit("0.9119800"), lit("0.3537500"), lit("1.8970500")
    fch3, hch3 = lit("0.0693700") * fact2, lit("0.1387400") * fact2
    fkinf, ak = lit("0.4291400") * fact2, lit("0.1353000") * fact2
    aa1, aa2, aa3, aa4 = lit("1.265960"), lit("0.000710"), lit("0.985920"), lit("2.785060")
    a1s, b1s = lit("1.5132681e-7"), lit("-4.3792246")
    a2s, b2s = lit("1.9202402e-7"), lit("-12.323018")
    pi = lit("3.141592653589793")
    anuzero = default_real("-0.70983241091685123")
    argmax = lit("19.0")

    q = [xi * lit("0.52918") for xi in x]
    atom = lambda a: q[3 * (a - 1): 3 * a]
    carbon, attacker = atom(2), atom(6)
    hydrogens = [atom(a) for a in (3, 4, 5, 1)]
    norm = lambda u: sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    tcb = [c - b for c, b in zip(carbon, attacker)]
    tch = [[c - h for c, h in zip(carbon, hy)] for hy in hydrogens]
    tbh = [[b - h for b, h in zip(attacker, hy)] for hy in hydrogens]
    rcb = norm(tcb)
    rch = [norm(t) for t in tch]
    rbh = [norm(t) for t in tbh]

    def value(v):
        return v.value() if isinstance(v, Series) else v

    s1, s2, s3, sphi, stheta = [], [], [], [], []
    for r in rch:
        for out, a, b, p in ((s1, a1s, b1s, 8), (s2, a2s, b2s, 6), (s3, a3s, b3s, 2)):
            arg = a * (r - r0ch) * (r - b) ** p
            out.append(1 - tanh(arg) if margins.below(value(arg), argmax, "saturation") else 0)
        inside = margins.below(value(r), lit("3.8"), "switch")
        for out, a, b, c in ((sphi, aphi, bphi, cphi), (stheta, atheta, btheta, ctheta)):
            out.append(1 - tanh(a * (r - r0ch) * exp(b * (r - c) ** 3)) if inside else 0)

    tau = acos(mpf(-1) / 3)
    halfpi = pi / 2
    twopi = 2 * pi
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    other = {p: tuple(m for m in range(4) if m not in p) for p in pairs}
    theta0 = {}
    for i, j in pairs:
        k, l = other[(i, j)]
        theta0[(i, j)] = theta0[(j, i)] = (
            tau
            + (tau - halfpi) * (sphi[i] * sphi[j] - 1)
            + (tau - twopi / 3) * (stheta[k] * stheta[l] - 1)
        )

    rav = (rch[0] + rch[1] + rch[2] + rch[3]) / 4
    arga = c1ch * (rav - r0ch)
    if margins.below(value(arga), argmax, "saturation"):
        ach = a1ch + b1ch * (tanh(arga) + 1) / 2
    else:
        ach = a1ch + b1ch

    def leps(d1, d3, a, r, r0):
        e1 = d1 * (exp(-2 * a * (r - r0)) - 2 * exp(-a * (r - r0)))
        e3 = d3 * (exp(-2 * a * (r - r0)) + 2 * exp(-a * (r - r0)))
        return (e1 + e3) / 2, (e1 - e3) / 2

    vqcb, vjcb = leps(d1cb, d3cb, acb, rcb, r0cb)
    vstr = 0
    for i in range(4):
        vqch, vjch = leps(d1ch, d3ch, ach, rch[i], r0ch)
        vqbh, vjbh = leps(d1hh, d3hh, ahh, rbh[i], r0hh)
        vj = -sqrt(((vjch - vjcb) ** 2 + (vjcb - vjbh) ** 2 + (vjbh - vjch) ** 2) / 2)
        vstr = vstr + vqch + vqcb + vqbh + vj

    def unit_bond(m):
        return [-t / rch[m] for t in tch[m]]

    def normal(j, k, l):
        a = [hk - hj for hk, hj in zip(hydrogens[k], hydrogens[j])]
        b = [hl - hj for hl, hj in zip(hydrogens[l], hydrogens[j])]
        axb = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        return axb, norm(axb)

    def dot(u, w):
        return u[0] * w[0] + u[1] * w[1] + u[2] * w[2]

    vop = 0
    for i in range(4):
        j, k, l = (i + 1) % 4, (i + 2) % 4, (i + 3) % 4
        axb, size = normal(j, k, l)
        for m in (j, k, l):
            if margins.above(value(dot(axb, unit_bond(m)) / size), 0, "handedness"):
                k, l = l, k
        axb, size = normal(j, k, l)
        sum2 = sum4 = 0
        for m in (j, k, l):
            delta = acos(dot(axb, unit_bond(m)) / size) - theta0[(i, m)]
            sum2 = sum2 + delta ** 2
            sum4 = sum4 + delta ** 4
        switch = 1 - s3[i]
        for m in range(4):
            if m != i:
                switch = switch * s3[m]
        vop = vop + switch * fch3 * sum2 + switch * hch3 * sum4

    f0 = fkinf + ak
    f2 = fkinf
    f1 = []
    for i in range(4):
        a1 = 1 - exp(-(aa1 * rbh[i] * rbh[i]))
        a2 = aa2 + aa3 * exp(-(aa4 * (rbh[i] - r0hh) * (rbh[i] - r0hh)))
        f1.append(a1 * exp(-a2 * (rch[i] - r0ch) ** 2))
    vip = 0
    for i, j in pairs:
        k, l = other[(i, j)]
        fk0 = f0 + f0 * (s1[i] * s1[j] - 1) + (f0 - f2) * (s2[k] * s2[l] - 1)
        cos = dot(tch[i], tch[j]) / rch[i] / rch[j]
        dth = acos(cos) - theta0[(i, j)]
        vip = vip + fk0 * f1[i] * f1[j] * dth ** 2 / 2

    return (vstr + vop + vip) * lit("0.03812") - anuzero


SERIES_OPS = (s_sqrt, s_exp, s_tanh, s_acos)
SCALAR_OPS = (mpmath.sqrt, mpmath.exp, mpmath.tanh, mpmath.acos)


def jet(x, v):
    margins = Margins()
    line = [Series([mpf(xi), mpf(vi)] + [mpf(0)] * (ORDER - 1)) for xi, vi in zip(x, v)]
    return energy(line, SERIES_OPS, margins).c, margins


def numerical_jet(x, v):
    margins = Margins()
    f = lambda t: energy([mpf(xi) + t * mpf(vi) for xi, vi in zip(x, v)], SCALAR_OPS, margins)
    return mpmath.taylor(f, 0, ORDER)


def low_orders(x, v):
    """E(x), g . v and v . H v from partial derivatives of the scalar energy.

    mpmath.diff differentiates the scalar transcription numerically at raised
    precision, coordinate by coordinate, so this shares no arithmetic with
    the power series that give the coefficients.
    """
    f = lambda *xs: energy(list(xs), SCALAR_OPS, Margins())
    point = [mpf(a) for a in x]

    def partial(*axes):
        orders = [0] * 18
        for a in axes:
            orders[a] += 1
        return mpmath.diff(f, point, orders)

    gradient = [partial(i) for i in range(18)]
    hessian = {}
    for i in range(18):
        for j in range(i, 18):
            hessian[(i, j)] = hessian[(j, i)] = partial(i, j)
    slope = sum(g * mpf(vi) for g, vi in zip(gradient, v))
    curvature = sum(
        mpf(v[i]) * hessian[(i, j)] * mpf(v[j]) for i in range(18) for j in range(18)
    )
    return f(*point), slope, curvature


def to_float(value, rounding):
    out = mpmath.libmp.to_float(mpf(value)._mpf_, rnd=rounding)
    assert (mpf(out) <= value) if rounding == "f" else (mpf(out) >= value)
    return out


def bracket(value):
    slack = abs(value) * mpf(10) ** (-(DPS - 15)) + mpf(10) ** (-(DPS - 10))
    return to_float(value - slack, "f"), to_float(value + slack, "c")


def cases():
    rng = random.Random(20261118)
    geometries = [("saddle", list(SADDLE))]
    for n in range(3):
        geometries.append(
            (f"displaced {n}", [a + rng.uniform(-0.15, 0.15) for a in SADDLE])
        )
    for label, x in geometries:
        for n in range(2):
            yield f"{label}, direction {n}", x, [rng.uniform(-1.0, 1.0) for _ in range(18)]


class Libcbe:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        d = ctypes.POINTER(ctypes.c_double)
        self.lib.cbe_eval.argtypes = [ctypes.c_int, d, d, d, ctypes.c_int]
        self.lib.cbe_hvp.argtypes = [
            ctypes.c_int, d, d, d, ctypes.POINTER(ctypes.c_int64), d,
        ]
        self.lib.cbe_init()
        self.lib.cbe_hvp_init()

    def evaluate(self, x, v):
        arr = ctypes.c_double * 18
        xs, vs, g, hv = arr(*x), arr(*v), arr(), arr()
        e = (ctypes.c_double * 1)()
        ops, scale = ctypes.c_int64(), ctypes.c_double()
        self.lib.cbe_eval(1, xs, e, g, 1)
        self.lib.cbe_hvp(1, xs, vs, hv, ctypes.byref(ops), ctypes.byref(scale))
        return e[0], list(g), list(hv), ops.value, scale.value


def floats(values):
    return " ".join(repr(float(a)) for a in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libcbe", required=True)
    parser.add_argument("--source", required=True, help="ch52008.f the library was built from")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.source, "rb") as fh:
        source = hashlib.sha256(fh.read()).hexdigest()
    lib = Libcbe(args.libcbe)
    out = open(args.output, "w")
    out.write("# CH5 2008 reference for rgpot-core/tests/ch5_2008.rs; written by generate.py.\n")
    out.write(f"# ch52008.f sha256 {source}; mpmath {mpmath.__version__} at {DPS} digits.\n")
    out.write(f"order {ORDER}\n")
    for label, x, v in cases():
        mp.dps = DPS
        coefficients, margins = jet(x, v)
        assert margins.saturation > 1 and margins.switch > mpf("0.1"), label
        assert margins.handedness > mpf("1e-3"), label
        mp.dps = CHECK_DPS
        numerical = numerical_jet(x, v)
        mp.dps = DPS
        for k, (a, b) in enumerate(zip(coefficients, numerical)):
            agree = abs(a - b) <= mpf(10) ** -25 * (1 + abs(a))
            assert agree, f"{label}: c_{k} {a} against mpmath.taylor {b}"
        independent = low_orders(x, v)
        series = (coefficients[0], coefficients[1], 2 * coefficients[2])
        for name, a, b in zip(("energy", "slope", "curvature"), independent, series):
            agree = abs(a - b) <= mpf(10) ** -30 * (1 + abs(a))
            assert agree, f"{label}: {name} {a} against the series {b}"
        e, g, hv, ops, scale = lib.evaluate(x, v)
        out.write(f"case {label}\n")
        out.write(f"x {floats(x)}\n")
        out.write(f"v {floats(v)}\n")
        for k, c in enumerate(coefficients):
            down, up = bracket(c)
            out.write(f"c {k} {down!r} {up!r}\n")
        for name, value in zip(("energy", "slope", "curvature"), independent):
            down, up = bracket(value)
            out.write(f"{name} {down!r} {up!r}\n")
        out.write(f"libcbe_energy {e!r}\n")
        out.write(f"libcbe_gradient {floats(g)}\n")
        out.write(f"libcbe_hvp {floats(hv)}\n")
        out.write(f"libcbe_hvp_roundoff {ops} {scale!r}\n")
        print(
            f"{label}: margins saturation {float(margins.saturation):.3g}, "
            f"switch {float(margins.switch):.3g}, handedness {float(margins.handedness):.3g}",
            file=sys.stderr,
        )
    out.close()


if __name__ == "__main__":
    main()
