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


def derive() -> None:
    """Pretty-print the q_v -> 1 limit of the Tsallis cooling schedule."""
    sp.init_printing(use_unicode=False)
    print("Theorem 4: logarithmic limit of Tsallis cooling schedule")
    print("  T_qv(t)        =", schedule)
    print("  limit q_v -> 1 =", limit)
    print("  target         =", target)
    print("  witness        =", WITNESS)


if __name__ == "__main__":
    derive()
