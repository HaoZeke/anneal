"""Theorem 1: The q_v -> 1 limit of the GSA visiting distribution is the
Boltzmann (Gaussian) form.

The GSA visiting distribution from manuscript Eq. (3) is parameterised by
q_v. Taking the limit q_v -> 1+ recovers T^{-D/2} * exp(-dx^2/T).

A negative-form variant is also tested: the literature's "numerator"
sign-convention writes the Tsallis bracket with a flipped sign, which
yields exp(+dx^2/T) at the limit -- a non-density. This module exposes
``witness_for(form)`` for the parametric negative test in tests/test_proofs.py.
"""

import sympy as sp

from proofs.helpers import witness

T, dx, q_v, D = sp.symbols("T dx q_v D", positive=True)


def g_qv_denominator(dx, T, q_v, D):
    """The correct Tsallis-bracket form (denominator-of-the-fraction)."""
    pref = T ** (-D / (3 - q_v))
    base = 1 + (q_v - 1) * dx**2 / T ** (2 / (3 - q_v))
    expo = -(1 / (q_v - 1) + (D - 1) / 2)
    return pref * base**expo


def g_qv_numerator(dx, T, q_v, D):
    """The literature's numerator-form (sign error): the exponent's leading
    minus sign is dropped, so the same `1 + (q_v-1) dx^2 / ...` bracket is
    raised to the positive power `+1/(q_v-1) + (D-1)/2`. At the q_v -> 1
    limit this gives exp(+dx^2/T), a non-density.
    """
    pref = T ** (-D / (3 - q_v))
    base = 1 + (q_v - 1) * dx**2 / T ** (2 / (3 - q_v))
    expo = 1 / (q_v - 1) + (D - 1) / 2
    return pref * base**expo


target = sp.exp(-(dx**2) / T) * T ** (-D / 2)


def witness_for(form: str) -> bool:
    """True iff the given Tsallis-bracket form matches the BSA Gaussian limit."""
    if form == "denominator":
        limit = sp.limit(g_qv_denominator(dx, T, q_v, D), q_v, 1, "+")
    elif form == "numerator":
        limit = sp.limit(g_qv_numerator(dx, T, q_v, D), q_v, 1, "+")
    else:
        raise ValueError(f"unknown form: {form}")
    return witness(limit, target)


WITNESS = witness_for("denominator")


def derive() -> None:
    """Pretty-print both forms' q_v -> 1 limit and the witness verdict."""
    sp.init_printing(use_unicode=False)
    g_d = g_qv_denominator(dx, T, q_v, D)
    g_n = g_qv_numerator(dx, T, q_v, D)
    print("Theorem 1: BSA Gaussian limit of GSA visiting distribution")
    print("  target           =", target)
    print("  denominator form (correct Tsallis bracket) =", g_d)
    print("    limit q_v->1+  =", sp.limit(g_d, q_v, 1, "+"))
    print("    witness (must be True)  =", witness_for("denominator"))
    print("  numerator form (sign-error negative test) =", g_n)
    print("    limit q_v->1+  =", sp.limit(g_n, q_v, 1, "+"))
    print("    witness (must be False) =", witness_for("numerator"))
    print("  WITNESS (module, denominator only) =", WITNESS)


if __name__ == "__main__":
    derive()
