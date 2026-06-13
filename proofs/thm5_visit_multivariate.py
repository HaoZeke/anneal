"""Theorem 5: the Tsallis visiting density (manuscript Eq. (3)) is the
isotropic D-dimensional multivariate Student-t, and is sampled as a Gaussian
scale mixture with a single shared mixing variable.

Eq. (3) reads (up to normalization, with r = |dx|^2)

    g_{q_v}(dx | T)  propto  [ 1 + (q_v - 1) r / T^{2/(3-q_v)} ]^{-(1/(q_v-1) + (D-1)/2)} .

Claim. With nu = (3 - q_v) / (q_v - 1) this is the density of a multivariate
Student-t with nu degrees of freedom and an isotropic scale s, i.e. the
Gaussian scale mixture

    x = scale * z / sqrt(g),   z ~ N(0, I_D),   g ~ Gamma(nu/2, 2/nu),

where g is drawn ONCE and shared across all D coordinates. The shared mixing
variable is what couples the coordinates into the isotropic joint law; a
per-coordinate g would give a product of 1-D Student-t marginals, which is not
Eq. (3) for D > 1.

This script verifies, symbolically:
  (E) the exponent identity  (D + nu)/2 = 1/(q_v-1) + (D-1)/2,
  (C) the bracket-coefficient identity  1/(nu s^2) = (q_v-1)/T^{2/(3-q_v)}
      for s^2 = T^{2/(3-q_v)} / (3 - q_v), and
  (M) that integrating the D-dim Gaussian kernel over g ~ Gamma(nu/2, 2/nu)
      reproduces the r-dependence of Eq. (3).
"""

import sympy as sp

from proofs.helpers import witness  # noqa: F401  (kept for parity with thmN scripts)

r, T, D, qv, g = sp.symbols("r T D q_v g", positive=True)
m, a = sp.symbols("m a", positive=True)

nu = (3 - qv) / (qv - 1)
s2 = T ** (sp.Integer(2) / (3 - qv)) / (3 - qv)

# (E) exponent / degrees-of-freedom identity.
EXPONENT_OK = sp.simplify((D + nu) / 2 - (sp.Integer(1) / (qv - 1) + (D - 1) * sp.Rational(1, 2))) == 0

# (C) bracket-coefficient identity: the multivariate-t base 1 + r/(nu s^2)
#     equals the displayed base 1 + (q_v - 1) r / T^{2/(3-q_v)}.
COEFF_OK = sp.simplify(1 / (nu * s2) - (qv - 1) / T ** (sp.Integer(2) / (3 - qv))) == 0

# (M) scale-mixture integral. The standard gamma integral
#     int_0^inf g^{m-1} e^{-a g} dg = Gamma(m) a^{-m} (m, a > 0) gives the
#     mixture density; substitute m = (D+nu)/2 and a = r/(2 s^2) + nu/2.
gamma_integral = sp.integrate(g ** (m - 1) * sp.exp(-a * g), (g, 0, sp.oo))  # Gamma(m) a^{-m}
mixed = gamma_integral.subs({m: (D + nu) / 2, a: r / (2 * s2) + nu / 2})

tsallis_kernel = (
    1 + (qv - 1) * r / T ** (sp.Integer(2) / (3 - qv))
) ** (-(sp.Integer(1) / (qv - 1) + (D - 1) * sp.Rational(1, 2)))


def _equal_up_to_r_constant(actual: sp.Expr, expected: sp.Expr) -> bool:
    """True iff actual/expected does not depend on r (equal up to an
    r-independent factor, the right notion for an unnormalized density in the
    sample variable)."""
    return sp.simplify(sp.diff(sp.log(actual / expected), r)) == 0


MIXTURE_OK = _equal_up_to_r_constant(mixed, tsallis_kernel)

WITNESS = bool(EXPONENT_OK and COEFF_OK and MIXTURE_OK)


def derive() -> None:
    """Pretty-print the multivariate-t identification of the visiting law."""
    sp.init_printing(use_unicode=False)
    print("Theorem 5: Tsallis visiting density = isotropic multivariate Student-t")
    print("  nu = (3 - q_v)/(q_v - 1) =", sp.simplify(nu))
    print("  exponent identity (E)    =", EXPONENT_OK)
    print("  coefficient identity (C) =", COEFF_OK)
    print("  scale-mixture (M)        =", MIXTURE_OK)
    print("  mixture r-kernel         =", sp.simplify(mixed))
    print("  Eq. (3) r-kernel         =", tsallis_kernel)
    print("  WITNESS                  =", WITNESS)


if __name__ == "__main__":
    derive()
