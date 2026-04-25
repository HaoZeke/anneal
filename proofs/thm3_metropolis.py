"""Theorem 3: The Q_a -> 1 limit of the GSA acceptance rule is the Metropolis
exp(-beta * dE) form.

The GSA acceptance from manuscript Eq. (5):
  P_Qa(dE, beta) = [1 - (1 - Q_a) * beta * dE]^{1 / (1 - Q_a)}
becomes exp(-beta * dE) as Q_a -> 1.
"""

import sympy as sp

from proofs.helpers import witness

beta, dE, Q_a = sp.symbols("beta dE Q_a", positive=True, real=True)
inner = (1 - (1 - Q_a) * beta * dE) ** (1 / (1 - Q_a))

# The min(1, ...) wrapper does not change the limit; both sides are bounded
# by 1 at dE >= 0 and unbounded otherwise. Take the limit on the inner.
limit = sp.limit(inner, Q_a, 1)

target = sp.exp(-beta * dE)

WITNESS = witness(limit, target)
