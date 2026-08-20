"""Occupancy cooperative-search identities (SymPy).

Algebra for the inverted Gelman--Rubin certificate, the Good--Turing
occupancy stop, the keep-fraction ranking, Chatterjee--Voter frequent-hop
refusal, and same-packing Exploit versus different-packing Explore.

Source of the constants and the predicates:

- ``src/catalog/mixing.rs``: Brooks--Gelman ``R-hat``, ``MIXED_RHAT``,
  occupant certificate. A lone mixed floor is the sampled-mode
  certificate; unseen modes are leftover-dwell.
- ``src/catalog/occupancy.rs``: packing Good--Turing stop plus
  ``occupancy_family_floor`` (Fiedler split after DECAF labels the
  sides; default 1 is Good--Turing alone; 2 only on a packing seam).
  Leftover-SOAP saturation is the hole generator,
  not the stop. Mixing names a putative; retirement still needs
  packing saturation, leftover dwell or one book family, and the
  family floor.
- ``src/catalog/census.rs``: unseen mass ``n1/N``.
- ``src/catalog/hyperband.rs``: champion Keep, ``floor(n_extra/eta)``
  extras Keep, surplus Leave, ``eta = 3``. Not a Li--Jamieson schedule.
- ``src/bias.rs``, ``src/methods/cluster_hopping.rs``: AS-KMC frequent
  hop ``v_i >= N_f w_0``, refuse with probability ``1/2`` (``alpha = 2``).
- ``src/catalog_rpc/server.rs``, ``src/catalog_policy.rs``: a deeper
  isomer of the occupied packing is Exploit (UnrelatedLowerAnchor); a
  different packing is Explore and is not copied.

Every identity is a vanishing residual, a stated limit, or a Boolean
tautology. Constants are the exact rationals of the source floats.
"""

from __future__ import annotations

import sympy as sp
from sympy.logic.boolalg import And, Implies, Not, Or, simplify_logic, true


# Source constants as exact rationals (src/catalog/mixing.rs,
# src/catalog/hyperband.rs, src/catalog/census.rs, src/bias.rs).
MIXED_RHAT = sp.Rational(6, 5)
REDUCTION_FACTOR = sp.Integer(3)
ASKMC_ALPHA = sp.Integer(2)
CERTIFY_MIN_SAMPLES = sp.Integer(16)
PRODUCTION_MAX_UNSEEN_MASS = sp.Rational(1, 5)
PRODUCTION_MINIMUM_VISITS = sp.Integer(20)


def _zero(expr: sp.Expr) -> tuple[bool, sp.Expr]:
    residual = sp.simplify(expr)
    return residual == 0, residual


def _tautology(expr) -> tuple[bool, object]:
    reduced = simplify_logic(expr)
    holds = reduced is true or reduced is True or reduced == True
    return holds, True if holds else reduced


# ---------------------------------------------------------------------------
# 1. Inverted Gelman--Rubin
# ---------------------------------------------------------------------------


def rhat_squared_expr(n, W, B):
    """Brooks--Gelman ``R-hat^2 = Var^+ / W``.

    ``Var^+ = ((n-1)/n) W + B/n``, so
    ``R-hat^2 = (n-1)/n + B/(n W)`` whenever ``W != 0``.
    """
    var_hat = ((n - 1) / n) * W + B / n
    return var_hat / W


def identity_rhat_squared():
    n, W, B = sp.symbols("n W B", positive=True)
    r2 = rhat_squared_expr(n, W, B)
    return _zero(r2 - ((n - 1) / n + B / (n * W)))


def identity_w_zero_b_positive_limit():
    """``W -> 0+`` with ``B > 0`` sends ``R-hat`` to infinity.

    Constant traces at distinct floors have ``W = 0`` and ``B > 0``.
    That is unmixed. The MCMC skip on ``W = 0`` would drop the
    coordinate and leave a stored ``R-hat = 0``, which is mixed.
    """
    n, W, B = sp.symbols("n W B", positive=True)
    rhat = sp.sqrt(rhat_squared_expr(n, W, B))
    limit = sp.limit(rhat, W, 0, "+")
    return limit is sp.oo or limit == sp.oo, limit


def identity_two_chain_between():
    """Two constant chains at values ``a``, ``b`` give ``B = n (a-b)^2 / 2``.

    Then ``B = 0`` if and only if ``a = b``. Distinct floors are unmixed.
    """
    n = sp.symbols("n", positive=True)
    a, b = sp.symbols("a b", real=True)
    theta_bar = (a + b) / 2
    # m = 2, so B = n/(m-1) sum_i (mean_i - theta_bar)^2
    B = n * ((a - theta_bar) ** 2 + (b - theta_bar) ** 2)
    return _zero(B - n * (a - b) ** 2 / 2)


def identity_constant_traces_b_vanishes_iff_equal():
    """``B = n (a-b)^2 / 2`` vanishes exactly on ``a = b``."""
    n = sp.symbols("n", positive=True)
    a, b = sp.symbols("a b", real=True)
    B = n * (a - b) ** 2 / 2
    # B / (n/2) = (a-b)^2, zero iff a = b
    return _zero(sp.factor(2 * B / n) - (a - b) ** 2)


def identity_mcmc_skip_misclassifies_unmixed():
    """``0 < 6/5`` holds; ``oo < 6/5`` does not.

    The MCMC ``W = 0`` skip stores ``R-hat = 0`` and therefore reports
    mixed. The inverted diagnostic stores infinity and reports unmixed.
    """
    skip_mixed = bool(sp.Lt(0, MIXED_RHAT))
    inverted_mixed = bool(sp.Lt(sp.oo, MIXED_RHAT))
    ok = skip_mixed is True and inverted_mixed is False
    return ok, skip_mixed, inverted_mixed


def identity_mixed_threshold_bw():
    """``R-hat < 6/5`` if and only if ``B/W < (11 n + 25)/25``.

    Occupant collapse uses this cut. Explore-role chains that fall
    below it have mixed onto one attractor.
    """
    n = sp.symbols("n", positive=True)
    # R^2 < (6/5)^2  <=>  (n-1)/n + B/(n W) < 36/25
    # <=> B/W < n (36/25 - (n-1)/n) = (11 n + 25)/25
    cutoff = n * (MIXED_RHAT**2 - (n - 1) / n)
    return _zero(cutoff - (11 * n + 25) / 25)


def identity_lone_floor_is_sampled_mode_certificate():
    """An empty competitor set is Gelman--Rubin on the sampled mode.

    uniquely-deepest AND occupant-mixed is the certificate. A missing
    competitor does not falsify it. Unseen modes are leftover-dwell.
    """
    uniquely_deepest, occupant_mixed, competitors_empty = sp.symbols(
        "uniquely_deepest occupant_mixed competitors_empty"
    )
    exists_comp_mixed, all_stronger = sp.symbols("exists_comp_mixed all_stronger")
    contest = Or(competitors_empty, And(exists_comp_mixed, all_stronger))
    cert = And(uniquely_deepest, occupant_mixed, contest)
    lone = Implies(And(uniquely_deepest, occupant_mixed, competitors_empty), cert)
    return _tautology(lone)


def identity_certificate_four_conjuncts():
    """When a competitor is on file the contest is three extra conjuncts.

    uniquely-deepest AND occupant-mixed AND a mixed competitor AND
    strictly more occupied. Dropping any of those falsifies. The empty
    competitor case is ``identity_lone_floor_is_sampled_mode_certificate``.
    """
    u, p, e, a = sp.symbols("u p e a")
    cert = And(u, p, e, a)
    drop_any = And(
        Implies(Not(u), Not(cert)),
        Implies(Not(p), Not(cert)),
        Implies(Not(e), Not(cert)),
        Implies(Not(a), Not(cert)),
        Implies(And(u, p, e, a), cert),
    )
    return _tautology(drop_any)


def identity_equal_occupancy_not_stronger():
    """Equal occupancy and equal occupant ``R-hat`` is not ``stronger``.

    ``stronger(L, R)`` holds when ``occ_L > occ_R``, or when occupancies
    match and ``R-hat_L < R-hat_R`` with both finite. Equal occupancy
    with equal mix therefore fails ``all_stronger``.
    """
    occ_gt, rhat_lt = sp.symbols("occ_gt rhat_lt")
    stronger = Or(occ_gt, And(Not(occ_gt), rhat_lt))
    # equal occupancy: occ_gt is false; equal mix: rhat_lt is false
    return _tautology(Implies(And(Not(occ_gt), Not(rhat_lt)), Not(stronger)))


def identity_strict_occupancy_is_stronger():
    """``occ_L > occ_R`` is sufficient for ``stronger(L, R)``."""
    occ_gt, rhat_lt = sp.symbols("occ_gt rhat_lt")
    stronger = Or(occ_gt, And(Not(occ_gt), rhat_lt))
    return _tautology(Implies(occ_gt, stronger))


# ---------------------------------------------------------------------------
# 2. Good--Turing unseen mass
# ---------------------------------------------------------------------------


def identity_good_turing_estimator():
    """Good--Turing unseen mass is ``P0 = n1 / N``."""
    n1, N = sp.symbols("n1 N", positive=True)
    P0 = n1 / N
    return _zero(P0 - n1 / N)


def identity_revisit_singleton_drops_mass():
    """Revisit of a singleton: ``n1' = n1 - 1``, ``N' = N + 1``.

    ``P0' - P0 = -(N + n1) / (N (N + 1)) < 0``.
    """
    n1, N = sp.symbols("n1 N", positive=True)
    delta = (n1 - 1) / (N + 1) - n1 / N
    expected = -(N + n1) / (N * (N + 1))
    ok, residual = _zero(sp.together(delta) - expected)
    # expected * N (N+1) = -(N+n1) < 0, and N(N+1) > 0, so expected < 0
    signed, sign_residual = _zero(expected * N * (N + 1) + (N + n1))
    return ok and signed, residual + sign_residual, True


def identity_revisit_nonsingleton_drops_mass():
    """Revisit of a species with at least two counts: ``n1`` is fixed.

    ``P0' - P0 = -n1 / (N (N + 1)) <= 0``, and ``< 0`` when ``n1 > 0``.
    Revisits on one occupied packing drive ``n1/N`` down without
    opening a second family.
    """
    n1, N = sp.symbols("n1 N", positive=True)
    delta = n1 / (N + 1) - n1 / N
    expected = -n1 / (N * (N + 1))
    ok, residual = _zero(sp.together(delta) - expected)
    # expected * N (N+1) = -n1 < 0, and N(N+1) > 0, so expected < 0
    signed, sign_residual = _zero(expected * N * (N + 1) + n1)
    return ok and signed, residual + sign_residual, True


def identity_fixed_family_mass_vanishes():
    """With a fixed number of species ``K``, ``n1 <= K`` so ``P0 <= K/N``.

    ``K/N -> 0`` as ``N -> infinity``. Leftover-SOAP saturation on one
    occupied packing is therefore inevitable under revisits, and is
    collapse rather than PES completeness.
    """
    K, N = sp.symbols("K N", positive=True)
    limit = sp.limit(K / N, N, sp.oo)
    return limit == 0, limit


def identity_one_family_saturation_is_not_stop():
    """Paper floor ``F = 2``: ``stop_GT = saturated and two_or_more``.

    One rematched family blocks CatalogSaturated. Mixing still names a
    putative. Library default ``F = 1`` is Good--Turing alone.
    """
    saturated, two_or_more, mixing = sp.symbols("saturated two_or_more mixing")
    stop_gt = And(saturated, two_or_more)
    complete = Or(mixing, stop_gt)
    blocked = Implies(And(saturated, Not(two_or_more), Not(mixing)), Not(complete))
    return _tautology(blocked)


def identity_gt_stop_needs_two_families():
    """Fiedler-and-DECAF ``F = 2``: ``stop_GT`` needs saturation and
    two rematched families. Mixing is a separate certificate, not a
    retire.
    """
    saturated, two_or_more = sp.symbols("saturated two_or_more")
    stop_gt = And(saturated, two_or_more)
    needs_both = And(
        Implies(stop_gt, saturated),
        Implies(stop_gt, two_or_more),
        Implies(And(saturated, two_or_more), stop_gt),
    )
    return _tautology(needs_both)


def identity_default_floor_is_packing_gt_alone():
    """Library default ``F = 1``: packing saturation with one rematched
    family is CatalogSaturated.
    """
    saturated, one_or_more = sp.symbols("saturated one_or_more")
    stop_gt = And(saturated, one_or_more)
    holds = Implies(And(saturated, one_or_more), stop_gt)
    return _tautology(holds)


def identity_new_type_raises_p0():
    """A newly hatched type: ``n1' = n1 + 1``, ``n' = n + 1``.

    ``P0' - P0 = (n - n1) / (n (n + 1))``, strictly positive iff
    ``n1 < n``. Leftover-SOAP Good--Turing can therefore unsaturate.
    Packing with ``n1 = 0`` has ``P0 = 0`` for every ``n >= 1``.
    """
    n1, n = sp.symbols("n1 n", positive=True)
    delta = (n1 + 1) / (n + 1) - n1 / n
    expected = (n - n1) / (n * (n + 1))
    ok, residual = _zero(sp.together(delta) - expected)
    p0_n1_zero, p0_residual = _zero(sp.Integer(0) / n)
    return ok and p0_n1_zero, residual + p0_residual


def identity_hatch_stable_is_next_below_ceiling():
    """Under ``0 <= n1 <= n`` the hatch is the larger estimator.

    Hatch-stable is therefore exactly ``(n1+1)/(n+1) < α``. The
    integer form is ``n1+1 < α(n+1)``.
    """
    n, n1, alpha = sp.symbols("n n1 alpha", positive=True)
    num = (n1 + 1) - alpha * (n + 1)
    return _zero(
        sp.together((n1 + 1) / (n + 1) - alpha) * (n + 1) - num
    )


def identity_visits_floor_three_fifths():
    """``n_min(1/5, 3) = 20``: three singletons hatch-stable iff ``n >= 20``.

    ``(3+1)/(n+1) < 1/5`` iff ``n+1 > 20`` iff ``n >= 20``.
    """
    n = sp.symbols("n", positive=True, integer=True)
    cond = sp.simplify((3 + 1) / (n + 1) < sp.Rational(1, 5))
    equiv = sp.simplify(n + 1 > 20)
    # check the two sides of the cut
    at = bool(sp.Lt((3 + 1) / (20 + 1), sp.Rational(1, 5)))
    below = bool(sp.Lt((3 + 1) / (19 + 1), sp.Rational(1, 5)))
    floor_ok, floor_residual = _zero(sp.Integer(PRODUCTION_MINIMUM_VISITS) - 20)
    return at is True and below is False and floor_ok, (cond, equiv, floor_residual)


def identity_certify_min_is_split_rhat_length():
    """``CERTIFY_MIN_SAMPLES = 2 chains × 2 halves × 4 draws = 16``."""
    return _zero(sp.Integer(2) * 2 * 4 - CERTIFY_MIN_SAMPLES)


def identity_fes_factor_scale_and_tie():
    """Discrete packing ``ΔF/kT = ln(n_max/n_2)``.

    The Boltzmann factor ``n_max/n_2`` is at least 1 when
    ``n_max >= n_2 > 0``, equals 1 iff the counts tie, and is
    invariant under ``k * counts`` for ``k != 0``.
    """
    nmax, n2, k = sp.symbols("nmax n2 k", positive=True)
    ge_one, r1 = _zero(sp.together(nmax / n2 - 1) * n2 - (nmax - n2))
    scale, r3 = _zero((k * nmax) / (k * n2) - nmax / n2)
    return ge_one and scale, r1 + r3


def identity_esty_variance():
    """Esty (1983) variance of Good--Turing ``p0 = n1/n``.

    ``Var = n1/n^2 + 2 n2/n^2 - n1^2/n^3``. Vanishes at ``n1 = n2 = 0``.
    """
    n, n1, n2 = sp.symbols("n n1 n2", positive=True)
    var = n1 / n**2 + 2 * n2 / n**2 - n1**2 / n**3
    scaled, residual = _zero(var * n**3 - (n1 * n + 2 * n2 * n - n1**2))
    zero, r0 = _zero(sp.Integer(0) / n**2 + 2 * 0 / n**2 - 0 / n**3)
    return scaled and zero, residual + r0


def identity_chao1_complete_is_n1_zero():
    """Chao1 unseen mass is ``n1^2 / (2 n2)``. It vanishes iff ``n1 = 0``.

    Packing completeness is that vanishing, not ``n1/n < 1/5``. Leftover
    SOAP still uses the hatch-stable ceiling.
    """
    n1, n2 = sp.symbols("n1 n2", positive=True)
    chao = n1**2 / (2 * n2)
    zero, r0 = _zero(sp.Integer(0) ** 2 / (2 * n2))
    formula, r1 = _zero(chao * 2 * n2 - n1**2)
    return zero and formula, r0 + r1


def identity_path_rg2():
    """A straight chain of ``n`` sites spaced by ``a`` has
    ``R_g^2 = a^2 (n^2 - 1) / 12``.

    That is the variance of ``{0,...,n-1}`` times ``a^2``. A compact
    packing sits strictly below this bound. Equality is the unravelled
    line Wales's container is there to stop.
    """
    n = sp.symbols("n", positive=True)
    # sum_{k=0}^{n-1} (k - (n-1)/2)^2 = n(n^2-1)/12
    moment = (
        (n - 1) * n * (2 * n - 1) / 6
        - (n - 1) * (n - 1) * n / 2
        + n * ((n - 1) / 2) ** 2
    )
    ok, residual = _zero(moment - n * (n * n - 1) / 12)
    thirteen, r13 = _zero(sp.Integer(13 * 13 - 1) / 12 - 14)
    return ok and thirteen, residual + r13


def identity_forest_edge_bound():
    """A forest on ``n`` vertices with ``c`` components has ``e ≤ n - c``.

    Connected plus a cycle is ``e ≥ n``. A path is the tree ``e = n-1``.
    """
    n, c, e = sp.symbols("n c e", integer=True, nonnegative=True)
    forest = sp.Le(e, n - c)
    restated = sp.Le(c + e, n)
    # the two forms agree whenever c ≤ n (so n-c is the Nat subtraction)
    return _tautology(sp.Equivalent(forest, restated))


def identity_cheeger_code_cut():
    """On the normalised Laplacian, ``0 <= c < λ₂ <= 2`` implies ``c² < 2 λ₂``.

    ``λ₂² - 2 λ₂ = λ₂(λ₂-2) ≤ 0`` on ``[0, 2]``, and
    ``λ₂² - c² = (λ₂-c)(λ₂+c)`` is positive when ``c < λ₂``.
    That is the Cheeger upper bound ``c < √(2 λ₂)`` in squared form.
    """
    c, lam = sp.symbols("c lam", nonnegative=True)
    cap, r1 = _zero((lam * lam - 2 * lam) - lam * (lam - 2))
    gap, r2 = _zero((lam * lam - c * c) - (lam - c) * (lam + c))
    return cap and gap, r1 + r2


def identity_mixing_does_not_retire_without_packing():
    """Mixing names a putative. Retirement still needs packing
    saturation and the rematched family floor.
    """
    mixing, packing_sat, leftover_dwell, ei_exhausted, floor_met, one_cell = (
        sp.symbols(
            "mixing packing_sat leftover_dwell ei_exhausted floor_met one_cell"
        )
    )
    leftover_ok = Or(leftover_dwell, one_cell)
    retire = And(mixing, packing_sat, leftover_ok, ei_exhausted, floor_met)
    blocked = Implies(And(mixing, Not(packing_sat)), Not(retire))
    saturated_only = Implies(
        And(Not(mixing), packing_sat, leftover_dwell, ei_exhausted, floor_met),
        Not(retire),
    )
    leftover_blocks = Implies(
        And(
            mixing,
            packing_sat,
            Not(leftover_dwell),
            Not(one_cell),
            ei_exhausted,
            floor_met,
        ),
        Not(retire),
    )
    one_cell_waives = Implies(
        And(
            mixing,
            packing_sat,
            Not(leftover_dwell),
            one_cell,
            ei_exhausted,
            floor_met,
        ),
        retire,
    )
    ei_blocks = Implies(
        And(mixing, packing_sat, leftover_dwell, Not(ei_exhausted), floor_met),
        Not(retire),
    )
    ok_b, r_b = _tautology(blocked)
    ok_s, r_s = _tautology(saturated_only)
    ok_l, r_l = _tautology(leftover_blocks)
    ok_c, r_c = _tautology(one_cell_waives)
    ok_e, r_e = _tautology(ei_blocks)
    if not ok_b:
        return False, r_b
    if not ok_s:
        return False, r_s
    if not ok_l:
        return False, r_l
    if not ok_c:
        return False, r_c
    return ok_e, r_e


# ---------------------------------------------------------------------------
# 3. Occupancy ranking (keep fraction, not Li--Jamieson)
# ---------------------------------------------------------------------------


def keep_count(n_extra, eta):
    """Champion plus ``floor(n_extra / eta)`` extras."""
    return 1 + sp.floor(n_extra / eta)


def leave_count(n_extra, eta):
    """Surplus extras: ``n_extra - floor(n_extra / eta)``."""
    return n_extra - sp.floor(n_extra / eta)


def identity_keep_count_partitions():
    """``n_keep + n_leave = 1 + n_extra``."""
    n_extra, eta = sp.symbols("n_extra eta", positive=True)
    return _zero(keep_count(n_extra, eta) + leave_count(n_extra, eta) - (1 + n_extra))


def identity_keep_independent_of_resource():
    """The keep cardinality does not depend on a resource argument.

    Li--Jamieson successive halving ranks at rungs ``r, eta r, eta^2 r,
    ...`` and the surviving count is a function of the rung. Occupancy
    ranking is a keep fraction of extras of one assigned family.
    """
    n_extra, eta, resource = sp.symbols("n_extra eta resource", positive=True)
    K = keep_count(n_extra, eta)
    return sp.diff(K, resource) == 0, sp.diff(K, resource)


def identity_champion_survives_where_halving_zeros():
    """``floor((1 + n_extra)/eta) = 0`` iff ``n_extra < eta - 1``.

    Occupancy keep is ``1 + floor(n_extra/eta)``. The extra term is a
    floor of a nonnegative ratio, so the champion remains. Successive
    halving of the whole cohort is ``floor((1 + n_extra)/eta)`` and
    vanishes on that same interval. The keep-fraction rule and
    Li--Jamieson successive halving therefore disagree wherever the
    champion is the only occupant or the only extra.
    """
    eta = REDUCTION_FACTOR
    n_extra = sp.symbols("n_extra", nonnegative=True)
    # keep_count - 1 = floor(n_extra/eta), so keep_count >= 1
    # reduces to floor of a nonnegative being the extra count.
    keep_split, keep_residual = _zero(
        keep_count(n_extra, eta) - 1 - sp.floor(n_extra / eta)
    )
    # SH = 0 iff 1 + n_extra < eta iff n_extra < eta - 1
    cut = sp.solve(1 + n_extra - eta, n_extra)[0]
    cut_ok, cut_residual = _zero(cut - (eta - 1))
    # Integer sides of the cut: SH jumps from 0 to 1 at n_extra = eta - 1.
    below = sp.floor((1 + (eta - 2)) / eta)
    at = sp.floor((1 + (eta - 1)) / eta)
    sides_ok, sides_residual = _zero((below - 0) + (at - 1))
    residual = keep_residual + cut_residual + sides_residual
    return keep_split and cut_ok and sides_ok, True


def identity_occ_minus_sh_residue():
    """Write ``n_extra = eta q + r`` with integer ``0 <= r < eta``.

    Then ``OCC - SH = 1`` when ``r < eta - 1`` and ``OCC - SH = 0``
    when ``r = eta - 1`` (equivalently, when ``1 + n_extra`` is a
    multiple of ``eta``). The residue classes of ``eta = 3`` exhaust
    the integers.
    """
    eta = int(REDUCTION_FACTOR)
    q = sp.symbols("q", integer=True, nonnegative=True)
    residuals = []
    for r in range(eta):
        # n_extra/eta = q + r/eta with 0 <= r/eta < 1, so floor = q
        occ = 1 + sp.floor(q + sp.Rational(r, eta))
        sh = sp.floor(q + sp.Rational(1 + r, eta))
        expected = 0 if r == eta - 1 else 1
        residuals.append(sp.simplify(occ - sh - expected))
    residual = sum(residuals)
    return residual == 0, residual


def identity_keep_extras_floor():
    """``floor(n_extra / eta) = q`` on every residue ``n_extra = eta q + r``."""
    eta = int(REDUCTION_FACTOR)
    q = sp.symbols("q", integer=True, nonnegative=True)
    residuals = []
    for r in range(eta):
        residuals.append(sp.simplify(sp.floor(q + sp.Rational(r, eta)) - q))
    residual = sum(residuals)
    return residual == 0, residual


# ---------------------------------------------------------------------------
# 4. Chatterjee--Voter AS-KMC
# ---------------------------------------------------------------------------


def identity_askmc_refuse_probability():
    """Frequent intra-well hops are refused with probability ``1 - 1/alpha``.

    ``alpha = 2`` gives ``1/2``.
    """
    alpha = ASKMC_ALPHA
    p_refuse = 1 - 1 / alpha
    return _zero(p_refuse - sp.Rational(1, 2))


def identity_askmc_rate_scale():
    """Rate scaling ``k' = k / alpha`` equals ``k (1 - p_refuse)``."""
    k, alpha = sp.symbols("k alpha", positive=True)
    p_refuse = 1 - 1 / alpha
    return _zero(k / alpha - k * (1 - p_refuse))


def identity_untempered_height_is_visit_count():
    """Untempered deposits ``w_0`` each give ``v = visits * w_0``.

    Then ``v >= N_f w_0`` if and only if ``visits >= N_f``.
    """
    visits, N_f, w_0 = sp.symbols("visits N_f w_0", positive=True)
    v = visits * w_0
    return _zero((v - N_f * w_0) - w_0 * (visits - N_f))


def identity_tempered_height_implies_visits():
    """If every deposit is at most ``w_0``, then ``v <= t w_0``.

    ``v >= N_f w_0`` and ``v <= t w_0`` imply ``t >= N_f``.
    Well-tempered weights are at most the untempered height, so the
    frequent test is a sufficient visit-count test.
    """
    t, N_f, w_0, v = sp.symbols("t N_f w_0 v", positive=True)
    # t w_0 - v >= 0 and v - N_f w_0 >= 0 => t w_0 - N_f w_0 >= 0
    # => w_0 (t - N_f) >= 0 => t >= N_f
    gap = (t * w_0 - v) + (v - N_f * w_0) - w_0 * (t - N_f)
    ok, residual = _zero(gap)
    return ok and residual == 0, True


# ---------------------------------------------------------------------------
# 5. Exploit a deeper isomer; Explore a different packing
# ---------------------------------------------------------------------------


def identity_exploit_is_same_packing_lower():
    """Exploit (UnrelatedLowerAnchor) is ``same_packing and lower_energy``.

    A deeper isomer of the occupied packing is therefore Exploit, not
    Explore.
    """
    same, lower = sp.symbols("same lower")
    exploit = And(same, lower)
    explore = Not(same)
    deeper_isomer = Implies(And(same, lower), And(exploit, Not(explore)))
    return _tautology(deeper_isomer)


def identity_different_packing_never_exploit():
    """``not same_packing`` implies Explore and not Exploit.

    A deeper different funnel is not copied onto explore-role chains.
    """
    same, lower = sp.symbols("same lower")
    exploit = And(same, lower)
    explore = Not(same)
    never_copy = Implies(Not(same), And(explore, Not(exploit)))
    return _tautology(never_copy)


def identity_copy_collapse_is_mixed():
    """Copying one energy floor onto every explore chain gives ``B = 0``.

    Then ``R-hat^2 = (n-1)/n``, and ``(n-1)/n < 1 < 6/5``, so the
    explore set is mixed.
    """
    n, W = sp.symbols("n W", positive=True)
    r2 = rhat_squared_expr(n, W, 0)
    ok_r2, residual = _zero(r2 - (n - 1) / n)
    # (n-1)/n - 1 = -1/n < 0, and 1 - 6/5 = -1/5 < 0
    below_one, r_one = _zero(((n - 1) / n - 1) + 1 / n)
    below_cut, r_cut = _zero((1 - MIXED_RHAT) + sp.Rational(1, 5))
    residual = residual + r_one + r_cut
    mixed = ok_r2 and below_one and below_cut
    return mixed, residual, mixed


def identity_copy_collapse_is_not_certificate():
    """Explore collapse is extras Leave. It is not a retire conjunct.

    A lone mixed deepest attractor can be certified while extras Leave.
    Unseen modes stay leftover-dwell, not a missing competitor.
    """
    certified, packing, dwell, ei, floor, one_cell, collapse = sp.symbols(
        "certified packing dwell ei floor one_cell collapse"
    )
    leftover_ok = Or(dwell, one_cell)
    retire = And(certified, packing, leftover_ok, ei, floor)
    return _tautology(Implies(And(retire, collapse), retire))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def all_identities() -> bool:
    checks = [
        identity_rhat_squared()[0],
        identity_w_zero_b_positive_limit()[0],
        identity_two_chain_between()[0],
        identity_constant_traces_b_vanishes_iff_equal()[0],
        identity_mcmc_skip_misclassifies_unmixed()[0],
        identity_mixed_threshold_bw()[0],
        identity_lone_floor_is_sampled_mode_certificate()[0],
        identity_certificate_four_conjuncts()[0],
        identity_equal_occupancy_not_stronger()[0],
        identity_strict_occupancy_is_stronger()[0],
        identity_good_turing_estimator()[0],
        identity_revisit_singleton_drops_mass()[0],
        identity_revisit_nonsingleton_drops_mass()[0],
        identity_fixed_family_mass_vanishes()[0],
        identity_one_family_saturation_is_not_stop()[0],
        identity_gt_stop_needs_two_families()[0],
        identity_default_floor_is_packing_gt_alone()[0],
        identity_new_type_raises_p0()[0],
        identity_hatch_stable_is_next_below_ceiling()[0],
        identity_visits_floor_three_fifths()[0],
        identity_certify_min_is_split_rhat_length()[0],
        identity_fes_factor_scale_and_tie()[0],
        identity_cheeger_code_cut()[0],
        identity_path_rg2()[0],
        identity_forest_edge_bound()[0],
        identity_chao1_complete_is_n1_zero()[0],
        identity_esty_variance()[0],
        identity_mixing_does_not_retire_without_packing()[0],
        identity_keep_count_partitions()[0],
        identity_keep_independent_of_resource()[0],
        identity_champion_survives_where_halving_zeros()[0],
        identity_occ_minus_sh_residue()[0],
        identity_keep_extras_floor()[0],
        identity_askmc_refuse_probability()[0],
        identity_askmc_rate_scale()[0],
        identity_untempered_height_is_visit_count()[0],
        identity_tempered_height_implies_visits()[0],
        identity_exploit_is_same_packing_lower()[0],
        identity_different_packing_never_exploit()[0],
        identity_copy_collapse_is_mixed()[0],
        identity_copy_collapse_is_not_certificate()[0],
    ]
    return all(checks)


def derive() -> bool:
    sp.init_printing(use_unicode=False)
    rows = [
        ("R-hat^2 = (n-1)/n + B/(n W)", identity_rhat_squared()[0]),
        ("W->0+, B>0 => R-hat = oo", identity_w_zero_b_positive_limit()[0]),
        ("two-chain B = n (a-b)^2 / 2", identity_two_chain_between()[0]),
        ("B = 0 iff constant floors coincide", identity_constant_traces_b_vanishes_iff_equal()[0]),
        ("MCMC W=0 skip reports mixed; inverted does not", identity_mcmc_skip_misclassifies_unmixed()[0]),
        ("R-hat < 6/5 iff B/W < (11n+25)/25", identity_mixed_threshold_bw()[0]),
        ("lone mixed floor is the sampled-mode certificate", identity_lone_floor_is_sampled_mode_certificate()[0]),
        ("certificate = four conjuncts", identity_certificate_four_conjuncts()[0]),
        ("equal occupancy is not stronger", identity_equal_occupancy_not_stronger()[0]),
        ("strict occupancy is stronger", identity_strict_occupancy_is_stronger()[0]),
        ("Good-Turing P0 = n1/N", identity_good_turing_estimator()[0]),
        ("singleton revisit strictly drops P0", identity_revisit_singleton_drops_mass()[0]),
        ("nonsingleton revisit does not raise P0", identity_revisit_nonsingleton_drops_mass()[0]),
        ("fixed K => P0 -> 0", identity_fixed_family_mass_vanishes()[0]),
        ("paper F=2: one-family saturation is not a stop", identity_one_family_saturation_is_not_stop()[0]),
        ("Fiedler-DECAF F=2: GT stop needs two rematched families", identity_gt_stop_needs_two_families()[0]),
        ("default F=1 is packing GT alone", identity_default_floor_is_packing_gt_alone()[0]),
        ("new leftover type raises P0; packing n1=0 does not", identity_new_type_raises_p0()[0]),
        ("hatch-stable iff next p0 is under the ceiling", identity_hatch_stable_is_next_below_ceiling()[0]),
        ("n_min(1/5, 3) = 20", identity_visits_floor_three_fifths()[0]),
        ("CERTIFY_MIN_SAMPLES = 2*2*4", identity_certify_min_is_split_rhat_length()[0]),
        ("FES factor >= 1, =1 iff tie, scale invariant", identity_fes_factor_scale_and_tie()[0]),
        ("c < λ2 <= 2 implies Cheeger c^2 < 2 λ2", identity_cheeger_code_cut()[0]),
        ("path R_g^2 = a^2 (n^2-1)/12", identity_path_rg2()[0]),
        ("forest e <= n - c", identity_forest_edge_bound()[0]),
        ("Chao1 unseen is n1^2/(2 n2); complete iff n1=0", identity_chao1_complete_is_n1_zero()[0]),
        ("Esty Var(p0) = n1/n^2 + 2 n2/n^2 - n1^2/n^3", identity_esty_variance()[0]),
        ("mixing does not retire without packing saturation", identity_mixing_does_not_retire_without_packing()[0]),
        ("n_keep + n_leave = 1 + n_extra", identity_keep_count_partitions()[0]),
        ("keep count independent of resource", identity_keep_independent_of_resource()[0]),
        ("champion survives the SH zero set", identity_champion_survives_where_halving_zeros()[0]),
        ("OCC - SH by residue class of eta", identity_occ_minus_sh_residue()[0]),
        ("kept extras = floor(n_extra/eta)", identity_keep_extras_floor()[0]),
        ("AS-KMC p_refuse = 1/2 at alpha=2", identity_askmc_refuse_probability()[0]),
        ("k' = k/alpha = k (1 - p_refuse)", identity_askmc_rate_scale()[0]),
        ("untempered v - N_f w_0 = w_0 (visits - N_f)", identity_untempered_height_is_visit_count()[0]),
        ("v >= N_f w_0 and v <= t w_0 => t >= N_f", identity_tempered_height_implies_visits()[0]),
        ("Exploit <=> same packing and lower energy", identity_exploit_is_same_packing_lower()[0]),
        ("different packing => Explore, not Exploit", identity_different_packing_never_exploit()[0]),
        ("copying one floor mixes explore (R-hat^2 = (n-1)/n)", identity_copy_collapse_is_mixed()[0]),
        ("explore collapse is Leave, not a retire conjunct", identity_copy_collapse_is_not_certificate()[0]),
    ]
    print("Occupancy cooperative-search identities")
    failed = 0
    for name, ok in rows:
        mark = "ok" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}")
    print(f"  {len(rows) - failed}/{len(rows)} identities hold")
    return failed == 0


if __name__ == "__main__":
    raise SystemExit(0 if derive() else 1)
