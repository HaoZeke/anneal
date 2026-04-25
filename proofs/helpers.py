"""Shared sympy setup and witness helper for the IISE-manuscript theorems.

Each theorem under proofs/thmN_*.py exposes a module-level WITNESS Boolean
that is True iff the theorem's symbolic identity holds. The witness helper
checks "equal up to a positive multiplicative constant" because the
manuscript identities are stated up to normalization.
"""

import sympy as sp


def witness(actual: sp.Expr, expected: sp.Expr) -> bool:
    """True iff ``actual`` and ``expected`` are equal up to a positive constant.

    Used by every thmN script to assert that two symbolic expressions agree
    (in `dx`, `T`, `D`, `beta`, `t` as appropriate) up to a positive scalar
    that does not depend on the integration variable.
    """
    ratio = sp.simplify(actual / expected)
    return ratio.is_constant() and bool(ratio > 0)
