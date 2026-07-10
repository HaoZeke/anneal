"""D5: Work-unit-optimal terminal polish reserve (budget endgame).

Model. A portfolio run holds an incumbent basin and a remaining budget of
B work units, spent on two activities:

  * Exploration slices of b work units. Each slice independently finds a
    strictly better basin with probability theta (stationary approximation,
    as in D4). A successful exploration replaces the incumbent, discarding
    any polish progress on the old basin.
  * Polish iterations of w work units on the incumbent. Polish contracts
    the relative gap g_k to the basin bottom geometrically:
    g_k = g_0 * rho**k with contraction factor 0 < rho < 1 (linear local
    convergence of projected gradient / trust-region refinement on a
    regular basin bottom).

Win criterion (matches the CUTEst protocol): final relative gap <= tau.

This module verifies, by symbolic algebra and exact enumeration:

  1. Closed-form polish requirement: the continuous switch point
     n_c = log(tau/g)/log(rho) satisfies g * rho**n_c = tau identically,
     and n* = ceil(n_c) is the minimal integer k with g * rho**k <= tau.
  2. Contraction estimator exactness: for any exactly geometric gap
     sequence g_k = c * rho**k, the estimator
     rho_hat = exp(mean_k log(g_{k+1}/g_k)) recovers rho exactly,
     independent of c; and the one-step improvement
     delta_k = g_k - g_{k+1} determines the residual gap through
     g_{k+1} = delta_k * rho/(1 - rho), so the pair (delta_last, rho_hat)
     estimates the unobservable gap without knowing the basin bottom.
  3. Explore-first is optimal (exchange argument): among all schedules
     with e exploration slices and p polish steps, the schedule that runs
     every exploration before any polish maximizes the win probability.
     Verified by exact symbolic enumeration of all C(e+p, e) schedules on
     small horizons: a polish step placed before the last exploration
     contributes only when every later exploration fails, so exchanging
     (P, E) -> (E, P) never lowers the win probability.
  4. Reserve-threshold optimality: with total slots n and the explore-first
     schedule form E^(n-p) P^p, the win probability is maximized at
     p = n*(g, tau, rho): fewer polish steps fail to convert the incumbent
     basin (gap stays above tau), while each extra polish step past n*
     forgoes exploration success probability at zero marginal polish value.
     Verified by exact symbolic evaluation of the win probability over all
     p on small horizons.

Style follows proofs/d4_thompson_allocation.py. Implementation mirror:
src/methods/portfolio.rs (endgame reserve and terminal polish phase).
"""

import itertools

import sympy as sp


# ---- Check 1: closed-form polish requirement --------------------------------
def polish_requirement_symbolic():
    g, tau, rho = sp.symbols("g tau rho", positive=True)
    n_c = sp.log(tau / g) / sp.log(rho)
    # Continuous switch point hits tau exactly: g * rho**n_c == tau.
    reaches = sp.simplify(g * rho**n_c - tau)
    ok_exact = reaches == 0
    # Minimality of the integer requirement on a numeric grid: n* = ceil(n_c)
    # is the least k with g rho^k <= tau (strictly above tau at k = n* - 1).
    ok_min = True
    for g0, t0, r0 in [(1, sp.Rational(1, 10**9), sp.Rational(1, 2)),
                       (100, sp.Rational(1, 10**6), sp.Rational(9, 10)),
                       (sp.Rational(3, 7), sp.Rational(1, 10**4), sp.Rational(1, 5))]:
        n_star = sp.ceiling(sp.log(t0 / g0) / sp.log(r0))
        ok_min &= bool(g0 * r0**n_star <= t0)
        if n_star >= 1:
            ok_min &= bool(g0 * r0 ** (n_star - 1) > t0)
    return ok_exact and ok_min


# ---- Check 2: contraction estimator exactness --------------------------------
def estimator_exactness_symbolic():
    c, rho = sp.symbols("c rho", positive=True)
    k, m = sp.symbols("k m", positive=True, integer=True)
    g = lambda i: c * rho**i
    # log-ratio estimator over m consecutive polish steps
    log_ratio = sp.log(g(k + 1) / g(k))
    rho_hat = sp.exp(sp.summation(log_ratio, (k, 0, m - 1)) / m)
    ok_rho = sp.simplify(rho_hat - rho) == 0
    # residual gap from the last one-step improvement:
    # delta_k = g_k - g_{k+1} = g_k (1 - rho)  =>  g_{k+1} = delta_k rho/(1-rho)
    delta_k = g(k) - g(k + 1)
    residual = delta_k * rho / (1 - rho)
    ok_gap = sp.simplify(residual - g(k + 1)) == 0
    return ok_rho and ok_gap


# ---- Check 3: explore-first schedule optimality (exact enumeration) ---------
def _win_probability(schedule, theta, n_star):
    """Exact symbolic win probability of a schedule over {E, P} slots.

    Two-basin model: the run starts in a bad basin whose bottom sits above
    the cell-best tolerance, so winning the cell requires (a) at least one
    successful exploration (each E slot finds the good basin independently
    with probability theta) and (b) at least n_star polish steps after the
    last successful exploration (a basin switch discards polish progress;
    n_star polish steps convert the fresh gap g to below tau)."""
    e_slots = [i for i, s in enumerate(schedule) if s == "E"]
    win = sp.Integer(0)
    for outcome in itertools.product([0, 1], repeat=len(e_slots)):
        if not any(outcome):
            continue  # never left the bad basin: loss
        prob = sp.Integer(1)
        last_success = -1
        for j, bit in enumerate(outcome):
            prob *= theta if bit else (1 - theta)
            if bit:
                last_success = e_slots[j]
        p_after = sum(1 for i, s in enumerate(schedule)
                      if s == "P" and i > last_success)
        if p_after >= n_star:
            win += prob
    return sp.simplify(win)


def explore_first_optimal_enumeration():
    theta = sp.Rational(1, 3)
    ok = True
    for e, p in [(2, 2), (3, 2), (2, 3)]:
        n_star = p  # the polish tail exactly converts the fresh gap
        slots = ["E"] * e + ["P"] * p
        best = None
        wins = {}
        for perm in set(itertools.permutations(slots)):
            w = _win_probability(perm, theta, n_star)
            wins[perm] = w
            if best is None or sp.simplify(w - best) > 0:
                best = w
        explore_first = tuple(["E"] * e + ["P"] * p)
        ok &= sp.simplify(wins[explore_first] - best) == 0
    return ok


# ---- Check 4: reserve-threshold optimality over p ----------------------------
def reserve_threshold_optimal_enumeration():
    theta = sp.Rational(1, 3)
    n = 5
    ok = True
    for n_star in (2, 3):
        win_at = {}
        for p in range(n + 1):
            schedule = tuple(["E"] * (n - p) + ["P"] * p)
            win_at[p] = _win_probability(schedule, theta, n_star)
        target = win_at[n_star]
        # p = n_star maximizes: fewer polish steps cannot convert the
        # incumbent; each extra polish step forgoes one exploration slot
        # at zero marginal polish value.
        for p, w in win_at.items():
            ok &= bool(sp.simplify(target - w) >= 0)
        ok &= bool(sp.simplify(target - win_at[n_star - 1]) > 0)
        ok &= bool(sp.simplify(target - win_at[min(n_star + 1, n)]) > 0)
    return ok


WITNESS_POLISH_REQUIREMENT = polish_requirement_symbolic()
WITNESS_ESTIMATOR = estimator_exactness_symbolic()
WITNESS_EXPLORE_FIRST = explore_first_optimal_enumeration()
WITNESS_RESERVE_THRESHOLD = reserve_threshold_optimal_enumeration()
WITNESS = (
    WITNESS_POLISH_REQUIREMENT
    and WITNESS_ESTIMATOR
    and WITNESS_EXPLORE_FIRST
    and WITNESS_RESERVE_THRESHOLD
)

if __name__ == "__main__":
    print("D5 polish requirement:", WITNESS_POLISH_REQUIREMENT)
    print("D5 estimator exactness:", WITNESS_ESTIMATOR)
    print("D5 explore-first optimal:", WITNESS_EXPLORE_FIRST)
    print("D5 reserve threshold optimal:", WITNESS_RESERVE_THRESHOLD)
    print("D5 WITNESS:", WITNESS)
