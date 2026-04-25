"""Theorem 4: The q_v -> 1 limit of the Tsallis cooling schedule is the
classical Boltzmann logarithmic schedule T_0 * log(2) / log(1 + t).

T_qv(t) = T_0 * (2^{q_v - 1} - 1) / ((1 + t)^{q_v - 1} - 1) -> log(2) / log(1+t)
"""

import sympy as sp

from proofs.helpers import witness

T0, t, q_v = sp.symbols("T0 t q_v", positive=True)
schedule = T0 * (2 ** (q_v - 1) - 1) / ((1 + t) ** (q_v - 1) - 1)
limit = sp.limit(schedule, q_v, 1)

target = T0 * sp.log(2) / sp.log(1 + t)

WITNESS = witness(limit, target)
