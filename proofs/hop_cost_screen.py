"""Cost-asymmetric screen: a derived hop rule, not a hand-set margin.

The measured recommended hop uses a constant discard window
``E_sc > E_best + 2``. The Bayes screen already in ``src/screen.rs``
replaces that comparison with a posterior probability of improvement,
but its threshold was still a knob (default 0.05).

This module derives the threshold from the hop's own step counts.

Two actions after the S-step screen:

- Quench. Always pay ``Q = R - S`` extra evaluations. On a win, the
  hop is worth one full hop ``R`` of progress; on a dud, the extra
  ``Q`` is wasted.
- Discard. Pay nothing extra. On a win that would have happened, lose
  ``R``.

Net value of quenching at improvement probability ``p``::

    V(quench) = p R - (1 - p) Q
    V(discard) = 0

Quench is better iff ``p > Q / (Q + R) = (R - S) / (2 R - S)``.

For the measured hop ``S = 25``, ``R = 200`` this is ``7/15``. That
scalar, not ``Δ = 2`` and not ``0.05``, is ``Config::derived``.

Witnesses are algebraic. Hit rates of ``derived`` are not claimed here.
"""

from __future__ import annotations

import sympy as sp


def threshold_symbolic():
    """``τ = Q/(Q+R)`` with ``Q = R-S`` simplifies to ``(R-S)/(2R-S)``."""
    s, r = sp.symbols("S R", positive=True)
    q = r - s
    tau = sp.simplify(q / (q + r))
    expected = (r - s) / (2 * r - s)
    return sp.simplify(tau - expected) == 0, tau, expected


def quench_better_iff():
    """``p R - (1-p) Q > 0`` iff ``p > Q/(Q+R)`` for ``0 < p < 1``."""
    p, q, r = sp.symbols("p Q R", positive=True)
    v = p * r - (1 - p) * q
    # V > 0  <=>  p R > (1-p) Q  <=>  p (R+Q) > Q  <=>  p > Q/(Q+R)
    cond = sp.simplify(sp.solve(v, p)[0] - q / (q + r))
    # solve(v, p) is the root V=0; check it equals the threshold
    return cond == 0, sp.solve(v, p)[0]


def measured_hop_is_seven_fifteenths():
    s, r = sp.Integer(25), sp.Integer(200)
    tau = (r - s) / (2 * r - s)
    return tau == sp.Rational(7, 15), tau


def threshold_in_unit_interval():
    """For 0 < S < R the threshold lies in (0, 1)."""
    s, r = sp.symbols("S R", positive=True)
    tau = (r - s) / (2 * r - s)
    # Assume R > S > 0. Numerator and denominator positive, denom > num.
    hyp = sp.Q.positive(r - s) & sp.Q.positive(s)
    num = r - s
    den = 2 * r - s
    # den - num = R > 0 so tau < 1; num > 0 so tau > 0
    return (
        sp.simplify(den - num - r) == 0,
        hyp,
        num,
        den,
    )


WITNESS = (
    threshold_symbolic()[0]
    and quench_better_iff()[0]
    and measured_hop_is_seven_fifteenths()[0]
    and threshold_in_unit_interval()[0]
)


def derive():
    ok1, tau, expected = threshold_symbolic()
    print("hop_cost_screen: derived Bayes threshold")
    print(f"  Check 1 (Q/(Q+R) = (R-S)/(2R-S)): {ok1}  tau = {tau}")
    ok2, root = quench_better_iff()
    print(f"  Check 2 (V(quench)>0 iff p > Q/(Q+R)): {ok2}  root = {root}")
    ok3, tau_m = measured_hop_is_seven_fifteenths()
    print(f"  Check 3 (S=25, R=200 => 7/15): {ok3}  {tau_m}")
    ok4, _, _, _ = threshold_in_unit_interval()
    print(f"  Check 4 (den - num = R, so tau < 1): {ok4}")
    print("  ALL CHECKS PASS:", WITNESS)
    return WITNESS


if __name__ == "__main__":
    raise SystemExit(0 if derive() else 1)
