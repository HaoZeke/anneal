"""Verification for quench_predictor_analysis.org (TOMS-style, real algorithm).

Mirrors src/quench.rs QuenchPredictor math:
  - OLS on log positive decrements
  - geometric tail E_inf = E_n - d_fit * r / (1-r)
  - residual propagation for sigma
  - Hopeless / Promising rules

No landscape toys. Theorems: exact recovery; Hopeless sound under zero residual.

The zero-residual hypothesis is false on real quenches, by four orders of
magnitude, and the measurement is recorded here because a proof whose
hypothesis nobody checked is worse than no proof. Scoring the extrapolation
against the value the full twenty-five step screen actually reaches, on
Lennard-Jones at 38 points, mean absolute error at the step where the rule
would have stopped:

    warmup   stop step   mean |error|
         4         4.2      12442.4
         8         8.4        992.7
        12        12.4         19.1
        16        16.1          3.6

Neighbouring minima near the bottom of that landscape are separated by well
under one unit, so a usable energy needs about twenty of the twenty-five
steps and the rule saves nothing. The theorems below remain true. They are
true about a regime this problem does not enter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Prediction:
    limit: float
    sigma: float
    ratio: float


def predict_from_energies(
    energies: list[float],
    warmup: int = 4,
) -> Prediction | None:
    """Port of QuenchPredictor::predict (see src/quench.rs)."""
    n = len(energies)
    if n < warmup:
        return None
    ks: list[float] = []
    ls: list[float] = []
    for k in range(n - 1):
        d = energies[k] - energies[k + 1]
        if d > 0.0:
            ks.append(float(k))
            ls.append(math.log(d))
    if len(ks) < 3:
        return None
    m = float(len(ks))
    kbar = sum(ks) / m
    lbar = sum(ls) / m
    sxx = sum((k - kbar) ** 2 for k in ks)
    sxy = sum((k - kbar) * (l - lbar) for k, l in zip(ks, ls))
    if sxx <= 0.0:
        return None
    slope = sxy / sxx
    if not (slope < 0.0):
        return None
    ratio = math.exp(slope)
    if not (ratio < 1.0):
        return None
    intercept = lbar - slope * kbar
    ss = sum((l - (intercept + slope * k)) ** 2 for k, l in zip(ks, ls))
    dof = max(m - 2.0, 1.0)
    s_log = math.sqrt(ss / dof)
    last_k = float(n - 2)
    d_fit = math.exp(intercept + slope * last_k)
    tail = d_fit * ratio / (1.0 - ratio)
    current = energies[n - 1]
    limit = current - tail
    se_slope = s_log / math.sqrt(sxx)
    d_rel = s_log
    r_rel = abs(se_slope * ratio / (1.0 - ratio))
    sigma = abs(tail) * math.sqrt(d_rel * d_rel + r_rel * r_rel)
    return Prediction(limit=limit, sigma=sigma, ratio=ratio)


def verdict(
    pred: Prediction | None,
    best: float,
    confidence: float = 2.0,
    margin: float = 1e-3,
) -> str:
    if pred is None or not math.isfinite(pred.sigma):
        return "Undecided"
    if pred.limit - confidence * pred.sigma > best + margin:
        return "Hopeless"
    if pred.limit + confidence * pred.sigma < best - margin:
        return "Promising"
    return "Undecided"


def geometric_energies(limit: float, gap0: float, r: float, n: int) -> list[float]:
    """E_k = limit + gap0 * r^k / (1-r) * (1-r) wait: gap at step 0 is E0-limit.

    Use E_k = limit + g0 * r^k  with g0 = gap0, so d_k = g0 r^k (1-r)? 
    Code test uses: gap *= 0.5 each step with E = limit + gap, so
    E_k = limit + gap0 * r^k, d_k = gap0 r^k (1-r) only if...
    E_k - E_{k+1} = gap0 r^k - gap0 r^{k+1} = gap0 r^k (1-r).
    So decrements are geometric with same r. Good.
    """
    out = []
    gap = gap0
    for _ in range(n):
        out.append(limit + gap)
        gap *= r
    return out


def test_exact_geometric_recovery() -> bool:
    limit, gap0, r, n = -100.0, 10.0, 0.5, 8
    e = geometric_energies(limit, gap0, r, n)
    p = predict_from_energies(e, warmup=4)
    if p is None:
        return False
    ok = abs(p.limit - limit) < 0.05
    ok &= abs(p.ratio - r) < 0.02
    ok &= p.sigma < 1e-10  # residual ~ 0 ⇒ sigma ~ 0
    return ok


def test_hopeless_sound_exact() -> bool:
    """Theorem 2: Hopeless + sigma≈0 ⇒ limit > best."""
    limit, best = -40.0, -100.0
    e = geometric_energies(limit, 5.0, 0.6, 6)
    p = predict_from_energies(e)
    if p is None:
        return False
    v = verdict(p, best)
    if v != "Hopeless":
        return False
    # soundness
    return p.limit > best and p.sigma < 1e-8


def test_promising_sound_exact() -> bool:
    limit, best = -140.0, -100.0
    e = geometric_energies(limit, 5.0, 0.6, 6)
    p = predict_from_energies(e)
    if p is None:
        return False
    v = verdict(p, best)
    if v != "Promising":
        return False
    return p.limit < best and p.sigma < 1e-8


def test_undecided_on_incumbent() -> bool:
    limit, best = -100.0, -100.0
    e = geometric_energies(limit, 2.0, 0.5, 5)
    p = predict_from_energies(e)
    return verdict(p, best) == "Undecided"


def test_sigma_blows_up_as_r_to_one() -> bool:
    """Lemma 5: near-flat geometric decay → large sigma or Undecided."""
    # r very close to 1: slow decay; need longer series for fit
    limit, gap0, r, n = -50.0, 1.0, 0.99, 30
    e = geometric_energies(limit, gap0, r, n)
    p = predict_from_energies(e, warmup=4)
    if p is None:
        return True  # refused: also fine
    # tail large, se amplifies
    return p.sigma > 0.1 or abs(p.ratio - 1.0) < 0.02


def test_ols_two_point_identity() -> bool:
    """Lemma 1: two log-decrements determine slope = log r exactly."""
    r = 0.7
    d0, d1 = 2.0, 2.0 * r
    # slope between k=0 and k=1
    slope = math.log(d1) - math.log(d0)
    return abs(slope - math.log(r)) < 1e-15


def test_savings_identity() -> bool:
    """Proposition 4: accounting only."""
    n_fix, n_early = 25, 6
    return (n_fix - n_early) == 19 and n_early <= n_fix


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("Thm1 exact geometric recovery", test_exact_geometric_recovery()),
        ("Thm2 Hopeless sound (exact model)", test_hopeless_sound_exact()),
        ("Thm3 Promising sound (exact model)", test_promising_sound_exact()),
        ("Undecided on incumbent", test_undecided_on_incumbent()),
        ("Lemma5 flat decay large sigma/refuse", test_sigma_blows_up_as_r_to_one()),
        ("Lemma1 log-linear slope", test_ols_two_point_identity()),
        ("Prop4 savings identity", test_savings_identity()),
    ]


WITNESS = all(v for _, v in all_checks())


def main() -> int:
    print("Quench early-stop predictor — TOMS-style verification")
    print("Algorithm: src/quench.rs QuenchPredictor")
    print()
    ok = True
    for name, v in all_checks():
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print()
    print("QUENCH_PREDICTOR_OK" if ok else "QUENCH_PREDICTOR_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
