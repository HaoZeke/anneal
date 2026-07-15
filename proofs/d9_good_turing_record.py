"""D9: Good-Turing × record discovery value.

Verifies the algebra behind BasinRegistry::record_discovery_prob:

  theta_disc = (n1 / n) / (w + 1)

and the record probability 1/(w+1) under exchangeability (exact for a
finite set of distinct depths).
"""

from __future__ import annotations

import itertools
import math

import sympy as sp


def discovery_value(n1: int, n: int, w: int) -> float:
    """D9.3: theta_disc = n1 / (n * (w + 1))."""
    if n <= 0 or w < 0 or n1 < 0:
        raise ValueError("invalid counts")
    if n1 > n or w > n:
        raise ValueError("inconsistent counts")
    return (n1 / n) / (w + 1)


def good_turing_missing_mass(n1: int, n: int) -> float:
    """D9.1: M_hat = n1 / n."""
    if n <= 0:
        raise ValueError("n must be positive")
    return n1 / n


def record_probability(w: int) -> float:
    """D9.2: P(new is best among w+1 exchangeable) = 1/(w+1)."""
    if w < 0:
        raise ValueError("w must be nonnegative")
    return 1.0 / (w + 1)


def symbolic_product_identity() -> bool:
    n1, n, w = sp.symbols("n1 n w", positive=True, integer=True)
    m_hat = n1 / n
    rec = 1 / (w + 1)
    theta = m_hat * rec
    return sp.simplify(theta - n1 / (n * (w + 1))) == 0


WITNESS = symbolic_product_identity()


def check_record_exchangeability(w: int = 4) -> bool:
    """Among w+1 distinct depths, each index is min equally often."""
    depths = list(range(w + 1))  # unique
    hits = [0] * (w + 1)
    for perm in itertools.permutations(depths):
        # last element is the "new" basin; first w are seen
        seen = perm[:w]
        new = perm[w]
        if new < min(seen):
            hits[w] += 1
        # also verify uniform min index
    total = math.factorial(w + 1)
    # fraction of perms where new is the overall min: (w)! / (w+1)! = 1/(w+1)
    return abs(hits[w] / total - 1.0 / (w + 1)) < 1e-15


def check_numeric_examples() -> bool:
    # Paper/code unit: n=6, w=3, n1=1 => (1/6)/4 = 1/24
    ok = abs(discovery_value(1, 6, 3) - (1.0 / 6.0) / 4.0) < 1e-15
    ok &= abs(discovery_value(0, 10, 5) - 0.0) < 1e-15
    ok &= abs(discovery_value(10, 10, 10) - (1.0) / 11.0) < 1e-15
    return ok


def check_monotone_in_singletons() -> bool:
    """More singletons => weakly larger discovery value at fixed n,w."""
    n, w = 20, 8
    prev = -1.0
    for n1 in range(0, n - w + 2):
        # feasible: at least w-n1 basins have >=1 if n1 <= w... keep simple
        if n1 > w:
            break
        # remaining n-n1 hits on w-n1 multi-hit basins need w-n1 >= 0
        multi = w - n1
        if multi > 0 and (n - n1) < multi:
            continue
        t = discovery_value(n1, n, w)
        if t + 1e-15 < prev:
            return False
        prev = t
    return True


def main() -> int:
    print("D9: Good-Turing × record discovery value")
    print(f"  WITNESS product identity: {WITNESS}")
    checks = [
        ("record exchangeability 1/(w+1)", check_record_exchangeability(5)),
        ("numeric registry examples", check_numeric_examples()),
        ("monotone in n1", check_monotone_in_singletons()),
    ]
    ok = WITNESS
    for name, v in checks:
        print(f"  {name}: {v}")
        ok = ok and v
    print(
        "  example n1=1 n=6 w=3 theta_disc=",
        f"{discovery_value(1, 6, 3):.6f}",
    )
    print("D9 OK" if ok else "D9 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
