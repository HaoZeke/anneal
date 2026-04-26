"""Theorem 2: At q_v = 2 the GSA visiting distribution specializes to a
Cauchy of the form g_2 propto T / (T^2 + dx^2)^{(D+1)/2}.

No limit needed: substitute q_v = 2 directly into the manuscript Eq. (3).
"""

import sympy as sp

from proofs.helpers import witness

T, dx, D = sp.symbols("T dx D", positive=True)
q_v = sp.Integer(2)

g = T ** (-D / (3 - q_v)) * (
    1 + (q_v - 1) * dx ** 2 / T ** (sp.Integer(2) / (3 - q_v))
) ** (-(sp.Integer(1) / (q_v - 1) + (D - 1) * sp.Rational(1, 2)))

target = T / (T ** 2 + dx ** 2) ** ((D + 1) * sp.Rational(1, 2))

WITNESS = witness(sp.simplify(g), target)


def derive() -> None:
    """Pretty-print the q_v = 2 substitution and the witness verdict."""
    sp.init_printing(use_unicode=False)
    print("Theorem 2: FSA Cauchy specialization of GSA visiting distribution")
    print("  q_v = 2 substitution =", sp.simplify(g))
    print("  target               =", target)
    print("  witness              =", WITNESS)


if __name__ == "__main__":
    derive()
