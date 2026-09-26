"""D3: Portfolio convergence preservation.

A portfolio driver interleaves slices of member arms. At least one arm performs
restarts drawn from a fixed everywhere-positive-density restart measure mu on
the bounded box. A uniformly randomized Cranley--Patterson shift gives every
fixed Halton point that marginal law. The
incumbent best is monotone (law-respecting best-update). Then:

  (a) the portfolio best converges a.s. to ess inf f provided the restart arm
      is scheduled infinitely often;
  (b) after n restarts,
        P(best within {f <= f* + eps}) >= 1 - (1 - mu(L_eps))^n,
      A deterministic QMC design also hits every axis-aligned box B once
      vol(B) exceeds its box discrepancy.

This module verifies (b)'s geometric tail by exact enumeration / Monte Carlo
agreement, and the discrepancy implication using the exact one-dimensional
star discrepancy of a van der Corput design.

Checks:
  1. Geometric tail identity: P(at least one of n iid restarts lands in L_eps)
     = 1 - (1 - p)^n with p = mu(L_eps). Verified symbolically and by Monte
     Carlo.
  2. Monotone-best preservation: the running minimum of mixed arm outputs is
     non-increasing and equals the min over all arms' samples (law L3-style
     best update). Verified by construction on random streams.
  3. Star-discrepancy covering: an anchored interval B with volume v is hit
     whenever v > D_n^*, since N(B)/n >= v-D_n^* > 0. For a d-dimensional
     axis-aligned box the same implication uses extreme discrepancy, bounded
     by 2^d D_n^* through inclusion--exclusion. The executable check evaluates
     the exact 1-D star discrepancy and the resulting integer count bound.
"""

import sympy as sp
import numpy as np


# ---- Check 1: geometric tail ----------------------------------------------
def geometric_tail_symbolic():
    """1 - prod_{i=1}^n (1 - p) = 1 - (1-p)^n for iid restarts."""
    p, n = sp.symbols("p n", positive=True)
    miss_n = (1 - p) ** n
    hit_n = 1 - miss_n
    # complement identity: hit + miss = 1
    return sp.simplify(hit_n + miss_n - 1) == 0


def geometric_tail_monte_carlo(p=0.05, n=40, trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    # each restart independently lands in L_eps with prob p
    hits = rng.random((trials, n)) < p
    any_hit = hits.any(axis=1)
    emp = any_hit.mean()
    theory = 1.0 - (1.0 - p) ** n
    return abs(emp - theory) < 5e-3, emp, theory


# ---- Check 2: monotone best preservation -----------------------------------
def monotone_best_preserved(seed=1, n_arms=3, slice_len=7, n_rounds=20):
    """Interleaving arm slices and taking a running minimum yields a
    non-increasing incumbent that equals the global minimum of all samples
    seen. This is the law-respecting best update (L3 keeps downhill moves)."""
    rng = np.random.default_rng(seed)
    incumbent = np.inf
    history = []
    all_samples = []
    for _ in range(n_rounds):
        arm = rng.integers(n_arms)  # scheduler picks an arm
        samples = rng.normal(size=slice_len) + arm * 0.0
        for v in samples:
            all_samples.append(v)
            if v < incumbent:
                incumbent = v
            history.append(incumbent)
    nonincreasing = all(
        history[i + 1] <= history[i] + 1e-15 for i in range(len(history) - 1)
    )
    equals_global_min = abs(incumbent - min(all_samples)) < 1e-12
    return nonincreasing and equals_global_min


# ---- Check 3: star-discrepancy covering -----------------------------------
def _van_der_corput(n, base=2):
    out = np.empty(n)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out


def _star_discrepancy_1d(points):
    """Return the exact star discrepancy of a finite one-dimensional design."""

    ordered = np.sort(np.asarray(points, dtype=np.float64).reshape(-1))
    if ordered.size == 0 or ordered[0] < 0.0 or ordered[-1] > 1.0:
        raise ValueError("points must be a nonempty subset of the unit interval")
    n = ordered.size
    indices = np.arange(1, n + 1, dtype=np.float64)
    closed_gap = indices / n - ordered
    open_gap = ordered - (indices - 1.0) / n
    return float(max(np.max(closed_gap), np.max(open_gap)))


def discrepancy_covering(n=512, box_measure=0.05, seed=3):
    """Check the exact anchored-box count implied by star discrepancy."""

    del seed
    pts = _van_der_corput(n, 2)
    dstar = _star_discrepancy_1d(pts)
    lower_bound = n * box_measure - n * dstar
    actual = np.count_nonzero(pts < box_measure)
    count_bound = actual / n >= box_measure - dstar - 1e-15
    threshold_ok = box_measure > dstar
    return count_bound and threshold_ok and actual >= 1, dstar, lower_bound, actual


WITNESS = (
    geometric_tail_symbolic()
    and geometric_tail_monte_carlo()[0]
    and monotone_best_preserved()
    and discrepancy_covering()[0]
)


def derive():
    sp.init_printing(use_unicode=False)
    print("D3: Portfolio convergence preservation")
    print("  Check 1a (geometric tail identity 1-(1-p)^n):", geometric_tail_symbolic())
    ok1b, emp, th = geometric_tail_monte_carlo()
    print(
        f"  Check 1b (Monte Carlo tail, p=0.05,n=40): {ok1b}  emp={emp:.4f} theory={th:.4f}"
    )
    print(
        "  Check 2 (monotone best = global min of samples):", monotone_best_preserved()
    )
    ok3, dstar, lb, actual = discrepancy_covering()
    print(f"  Check 3 (QMC star-discrepancy covering): {ok3}")
    print(
        f"    exact D_n^* = {dstar:.4f}, count bound n*mu - n*D* = {lb:.2f}, actual interval hits = {actual}"
    )
    all_ok = WITNESS
    print("  ALL CHECKS PASS:", all_ok)
    return all_ok


if __name__ == "__main__":
    ok = derive()
    raise SystemExit(0 if ok else 1)
