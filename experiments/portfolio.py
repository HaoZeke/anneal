"""Thompson-allocated portfolio over anneal building blocks.

The driver treats each building block (QMC-seeded multistart descent,
adaptive basin hopping, preconditioned GLE-Langevin, best/1/bin
differential evolution, generalized simulated annealing, additive
surrogate independence proposals, and shifted-QMC trust-region polls)
as an arm of a Bernoulli bandit. A discounted Beta-Bernoulli posterior
tracks the probability that one budget slice of an arm improves the
incumbent; Thompson sampling allocates the next slice. A probability
floor on the QMC restart arm keeps the restart measure scheduled
infinitely often, which preserves the global convergence guarantee of
the restart arm regardless of how the posterior concentrates.

Every objective and native-gradient evaluation is charged through the
shared budget counter; surrogate fits reuse archived evaluations at
zero marginal cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

import numpy as np
from scipy.optimize import minimize

from .tensor_surrogate import AdditiveSurrogate


def _anneal_module():
    return import_module("anneal")


def _is_budget_exc(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "_Budget"


def _best_finite(*values: float) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return min(finite) if finite else float("inf")


@dataclass(frozen=True)
class PortfolioConfig:
    """Tuning surface for the Thompson portfolio driver."""

    slice_divisor: int = 40
    slice_dim_multiplier: int = 8
    slice_min: int = 32
    restart_floor: float = 0.12
    improvement_rtol: float = 1e-4
    improvement_atol: float = 1e-12
    discount: float = 0.97
    prior_success: float = 1.0
    prior_failure: float = 1.0
    final_polish_fraction: float = 0.06
    final_polish_min: int = 50
    archive_cap: int = 8192
    explore_eval_fraction: float = 0.35
    hop_initial_step: float = 0.25
    hop_step_grow: float = 1.3
    hop_step_shrink: float = 0.75
    hop_step_min: float = 1e-4
    hop_step_max: float = 1.0
    surrogate_degree: int = 8
    surrogate_min_archive: int = 64
    surrogate_refit_every: int = 1
    gle_min_dim: int = 2
    gle_dt: float = 0.2
    gle_n_epochs: int = 40
    de_pop_min: int = 16
    de_pop_dim_multiplier: int = 4
    de_pop_max: int = 48
    de_weight_min: float = 0.5
    de_weight_span: float = 0.5
    de_crossover: float = 0.7
    gsa_t_init: float = 1.0
    gsa_q_v: float = 2.62
    gsa_q_a: float = 1.7
    tr_levels: int = 3
    metropolis_floor: float = 1e-12
    native_bounds_slack: float = 1e-9


class _Recorder:
    """Counter proxy that archives every objective evaluation."""

    def __init__(self, counter, cap: int):
        self._counter = counter
        self._cap = int(cap)
        self.xs: list[np.ndarray] = []
        self.fs: list[float] = []
        self.best_x: np.ndarray | None = None

    # -- counter protocol -------------------------------------------------
    @property
    def budget(self) -> int:
        return self._counter.budget

    @budget.setter
    def budget(self, value: int) -> None:
        self._counter.budget = value

    @property
    def n(self) -> int:
        return self._counter.n

    @property
    def best(self) -> float:
        return self._counter.best

    @best.setter
    def best(self, value: float) -> None:
        self._counter.best = value

    def counted_grad(self, grad):
        counted = getattr(self._counter, "counted_grad", None)
        return counted(grad) if callable(counted) and grad is not None else None

    def __call__(self, x):
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        prior_best = self._counter.best
        value = float(self._counter(x_arr))
        if len(self.xs) < self._cap:
            self.xs.append(x_arr.copy())
            self.fs.append(value)
        if math.isfinite(value) and value < prior_best:
            self.best_x = x_arr.copy()
        return value


class _ArmPosterior:
    """Discounted Beta-Bernoulli posterior over slice improvements."""

    def __init__(self, config: PortfolioConfig):
        self.alpha = config.prior_success
        self.beta = config.prior_failure
        self.discount = config.discount
        self.pulls = 0

    def update(self, success: bool) -> None:
        self.alpha = 1.0 + self.discount * (self.alpha - 1.0)
        self.beta = 1.0 + self.discount * (self.beta - 1.0)
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.pulls += 1

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta))


def _slice_budget(budget: int, dim: int, config: PortfolioConfig) -> int:
    return int(
        max(
            config.slice_min,
            config.slice_dim_multiplier * (dim + 1),
            budget // config.slice_divisor,
        )
    )


def _metropolis(delta: float, temp: float, rng, *, floor: float) -> bool:
    if delta <= 0.0:
        return True
    if not math.isfinite(delta):
        return False
    return bool(rng.random() < math.exp(-delta / max(temp, floor)))


def _temperature(state: dict, gen: int) -> float:
    t0 = state.get("temp0", 1.0)
    return max(t0 * math.log(2.0) / math.log(gen + 2.0), 1e-9)


def _incumbent(rec: _Recorder, low, high):
    if rec.best_x is not None:
        return np.clip(rec.best_x, low, high)
    return 0.5 * (np.asarray(low) + np.asarray(high))


def _lbfgs(rec: _Recorder, jac, x0, bounds, maxfun: int):
    if maxfun < 2:
        return None
    return minimize(
        rec,
        np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
        jac=jac,
        bounds=bounds,
        options={"maxfun": maxfun, "maxiter": maxfun},
    )


# -- arms -----------------------------------------------------------------
#
# Each arm consumes the recorder budget up to a temporary ceiling set by
# the scheduler; persistent per-arm state lives in ``state``.


def _arm_explore(ctx, state, slice_budget: int) -> None:
    """QMC restart arm: CP-shifted Halton starts ranked, best one polished.

    This is the floor-protected restart arm; its restart measure has
    uniform marginals over the box.
    """
    rec, low, high, dim = ctx["rec"], ctx["low"], ctx["high"], ctx["dim"]
    rng, config = ctx["rng"], ctx["config"]
    anneal = _anneal_module()
    skip = state.setdefault("skip", 1)
    n_starts = max(4, int(slice_budget * config.explore_eval_fraction))
    points = np.asarray(
        anneal.low_discrepancy_points(low, high, n_starts, int(skip)),
        dtype=np.float64,
    )
    state["skip"] = skip + n_starts
    width = high - low
    active = width > 0.0
    if np.any(active):
        unit = np.zeros_like(points)
        unit[:, active] = (points[:, active] - low[active]) / width[active]
        unit[:, active] = (unit[:, active] + rng.random(int(active.sum()))) % 1.0
        points = low + width * unit
    best_v, best_x = float("inf"), None
    for point in points:
        value = rec(point)
        if math.isfinite(value) and value < best_v:
            best_v, best_x = value, point
    if best_x is None:
        return
    remaining = rec.budget - rec.n
    if ctx["jac"] is not None and remaining >= 4:
        _lbfgs(rec, ctx["jac"], best_x, ctx["bounds"], remaining // 2)
    elif remaining >= 4:
        anneal.qmc_gsa_global_search(
            rec, low, high, remaining,
            seed=int(rng.integers(1 << 31)),
            n_chains=max(2, min(dim, remaining // 4)),
            t_init=config.gsa_t_init, q_v=config.gsa_q_v, q_a=config.gsa_q_a,
        )


def _arm_hop(ctx, state, slice_budget: int) -> None:
    """Adaptive-step basin hop around the incumbent with polished accepts."""
    rec, low, high, dim = ctx["rec"], ctx["low"], ctx["high"], ctx["dim"]
    rng, config = ctx["rng"], ctx["config"]
    step = state.setdefault("step", config.hop_initial_step)
    x_cur = state.get("x_cur")
    if x_cur is None:
        x_cur = _incumbent(rec, low, high)
    f_cur = state.get("f_cur", rec.best)
    width = np.where(high > low, high - low, 1.0)
    n_hops = 3
    per_hop = max(4, slice_budget // n_hops)
    temp = _temperature(state, state.setdefault("gen", 0))
    for _ in range(n_hops):
        if rec.n + 4 > rec.budget:
            break
        trial = np.clip(x_cur + rng.normal(0.0, step, dim) * width, low, high)
        if ctx["jac"] is not None:
            res = _lbfgs(rec, ctx["jac"], trial, ctx["bounds"], per_hop // 2)
            if res is None:
                break
            f_new = float(res.fun)
            polished = np.asarray(res.x, dtype=np.float64)
        else:
            f_new = rec(trial)
            polished = trial
        if not math.isfinite(f_new):
            step = max(step * config.hop_step_shrink, config.hop_step_min)
            continue
        if f_new < f_cur or _metropolis(
            f_new - f_cur, temp, rng, floor=config.metropolis_floor
        ):
            x_cur, f_cur = polished, f_new
            step = min(step * config.hop_step_grow, config.hop_step_max)
        else:
            step = max(step * config.hop_step_shrink, config.hop_step_min)
    state["step"] = step
    state["x_cur"], state["f_cur"] = x_cur, f_cur
    state["gen"] = state["gen"] + 1


def _arm_gle(ctx, state, slice_budget: int) -> None:
    """Preconditioned GLE-Langevin segment seeded at the incumbent."""
    rec, low, high = ctx["rec"], ctx["low"], ctx["high"]
    rng, config = ctx["rng"], ctx["config"]
    grad_fn = ctx["grad"]
    if grad_fn is None:
        return
    jac = rec.counted_grad(grad_fn)
    maxf = slice_budget // 2
    if maxf < 4:
        return
    anneal = _anneal_module()
    gle = getattr(anneal, "gle_langevin_preconditioned", None) or anneal.gle_langevin
    gle(
        rec, jac, low, high,
        max_fevals=maxf,
        seed=int(rng.integers(1 << 31)),
        dt=config.gle_dt,
        n_epochs=config.gle_n_epochs,
        x0=_incumbent(rec, low, high),
    )


def _arm_de(ctx, state, slice_budget: int) -> None:
    """Persistent best/1/bin differential evolution population."""
    rec, low, high, dim = ctx["rec"], ctx["low"], ctx["high"], ctx["dim"]
    rng, config = ctx["rng"], ctx["config"]
    pop = state.get("pop")
    if pop is None:
        anneal = _anneal_module()
        pop_size = int(
            min(
                config.de_pop_max,
                max(config.de_pop_min, config.de_pop_dim_multiplier * dim),
            )
        )
        points = np.asarray(
            anneal.low_discrepancy_points(low, high, pop_size, 1),
            dtype=np.float64,
        )
        pop = [p.copy() for p in points]
        vals = []
        for p in pop:
            if rec.n >= rec.budget:
                break
            vals.append(rec(p))
        pop = pop[: len(vals)]
        state["pop"], state["vals"] = pop, np.asarray(vals, dtype=np.float64)
        if len(pop) < 4:
            state["pop"] = None
            return
    pop, vals = state["pop"], state["vals"]
    finite = np.flatnonzero(np.isfinite(vals))
    if finite.size == 0:
        return
    best_i = int(finite[np.argmin(vals[finite])])
    best_x, best_v = pop[best_i].copy(), float(vals[best_i])
    used = 0
    while used < slice_budget and rec.n < rec.budget:
        weight = config.de_weight_min + config.de_weight_span * rng.random()
        for i in range(len(pop)):
            if used >= slice_budget or rec.n >= rec.budget:
                break
            others = [j for j in range(len(pop)) if j != i]
            r0, r1 = rng.choice(others, 2, replace=False)
            mutant = best_x + weight * (pop[int(r0)] - pop[int(r1)])
            mask = rng.random(dim) < config.de_crossover
            mask[rng.integers(dim)] = True
            trial = np.clip(np.where(mask, mutant, pop[i]), low, high)
            ft = rec(trial)
            used += 1
            if math.isfinite(ft) and (
                not math.isfinite(float(vals[i])) or ft < float(vals[i])
            ):
                pop[i], vals[i] = trial, ft
                if ft < best_v:
                    best_v, best_x = ft, trial.copy()
    state["vals"] = vals


def _arm_gsa(ctx, state, slice_budget: int) -> None:
    """Generalized simulated annealing slice with QMC-seeded chains."""
    rec, low, high, dim = ctx["rec"], ctx["low"], ctx["high"], ctx["dim"]
    rng, config = ctx["rng"], ctx["config"]
    if slice_budget < 8:
        return
    anneal = _anneal_module()
    anneal.qmc_gsa_global_search(
        rec, low, high, slice_budget,
        seed=int(rng.integers(1 << 31)),
        n_chains=max(2, min(4 * dim, slice_budget // 8)),
        t_init=config.gsa_t_init,
        q_v=config.gsa_q_v,
        q_a=config.gsa_q_a,
    )


def _arm_surrogate(ctx, state, slice_budget: int) -> None:
    """Additive surrogate independence proposals fit from the archive.

    The fit reuses archived evaluations, so the slice spends budget only
    on scoring proposals against the true objective; the acceptance
    probability of the tempered independence chain is bounded below in
    the sup-norm surrogate error.
    """
    rec, low, high, dim = ctx["rec"], ctx["low"], ctx["high"], ctx["dim"]
    rng, config = ctx["rng"], ctx["config"]
    if len(rec.fs) < max(config.surrogate_min_archive, 4 * dim):
        return
    X = np.asarray(rec.xs, dtype=np.float64)
    y = np.asarray(rec.fs, dtype=np.float64)
    keep = np.isfinite(y)
    if keep.sum() < max(config.surrogate_min_archive, 4 * dim):
        return
    surr = AdditiveSurrogate.from_points(
        X[keep], y[keep], low, high, degree=config.surrogate_degree
    )
    gen = state.setdefault("gen", 0)
    finite = y[keep]
    state.setdefault("temp0", max(float(np.std(finite)), 1e-6))
    # The modal point (per-coordinate argmin, the T -> 0 limit of the
    # tempered marginals) tests the surrogate's global candidate at the
    # cost of one evaluation; for separable objectives it is the global
    # minimizer once the fit settles.
    modal = np.empty(dim)
    for j in range(dim):
        xs_j, g_j = surr._coord_grid_energy(j, 65)
        modal[j] = xs_j[int(np.argmin(g_j))]
    before_modal = rec.best
    modal_val = rec(np.clip(modal, low, high))
    if (
        ctx["jac"] is not None
        and math.isfinite(modal_val)
        and modal_val < before_modal
        and rec.n + 4 <= rec.budget
    ):
        _lbfgs(rec, ctx["jac"], modal, ctx["bounds"], (rec.budget - rec.n) // 2)
    # Cool with budget progress so the ladder reaches the cold regime
    # regardless of how often the arm is pulled.
    total = ctx.get("total_budget", rec.budget)
    progress = rec.n / max(total, 1)
    temp = max(state["temp0"] * 0.5 ** (int(12.0 * progress) + gen), 1e-12)
    f_cur = rec.best
    proposals = surr.sample(slice_budget, temp, rng)
    for trial in proposals:
        if rec.n >= rec.budget:
            break
        ft = rec(np.clip(trial, low, high))
        if math.isfinite(ft) and _metropolis(
            ft - f_cur, temp, rng, floor=config.metropolis_floor
        ):
            f_cur = ft
    state["gen"] = gen + 1


def _arm_tr_poll(ctx, state, slice_budget: int) -> None:
    """Shifted-QMC trust-region poll around the incumbent."""
    rec, low, high = ctx["rec"], ctx["low"], ctx["high"]
    rng, config = ctx["rng"], ctx["config"]
    if slice_budget < 8:
        return
    anneal = _anneal_module()
    anneal.qmc_trust_region_poll(
        rec,
        low,
        high,
        _incumbent(rec, low, high),
        slice_budget,
        seed=int(rng.integers(1 << 31)),
        n_levels=config.tr_levels,
    )


_ARMS = {
    "explore": _arm_explore,
    "hop": _arm_hop,
    "gle": _arm_gle,
    "de": _arm_de,
    "gsa": _arm_gsa,
    "surrogate": _arm_surrogate,
    "tr_poll": _arm_tr_poll,
}
RESTART_ARM = "explore"


def _enabled_arms(dim: int, grad, config: PortfolioConfig) -> list[str]:
    arms = ["explore", "de", "gsa", "surrogate", "tr_poll"]
    if grad is not None:
        arms.insert(1, "hop")
        if dim >= config.gle_min_dim:
            arms.insert(3, "gle")
    return arms


def thompson_portfolio(
    counter,
    low,
    high,
    dim: int,
    grad,
    rng: np.random.Generator,
    *,
    config: PortfolioConfig | None = None,
    anchor=None,
):
    """Posterior-driven portfolio over anneal building blocks."""
    config = PortfolioConfig() if config is None else config
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    rec = _Recorder(counter, config.archive_cap)
    bounds = list(zip(low, high))
    jac = rec.counted_grad(grad)
    ctx = {
        "rec": rec, "low": low, "high": high, "dim": int(dim),
        "rng": rng, "config": config, "jac": jac, "grad": grad,
        "bounds": bounds, "total_budget": counter.budget,
    }
    total_budget = counter.budget
    final_polish = (
        max(config.final_polish_min, int(config.final_polish_fraction * total_budget))
        if jac is not None
        else 0
    )
    main_ceiling = max(total_budget - final_polish, counter.n)

    if anchor is not None:
        anchor_arr = np.clip(
            np.asarray(anchor, dtype=np.float64).reshape(-1), low, high
        )
        try:
            rec(anchor_arr)
        except Exception as exc:  # noqa: BLE001
            if not _is_budget_exc(exc):
                raise
            return _best_finite(counter.best)

    arms = _enabled_arms(dim, grad, config)
    posteriors = {name: _ArmPosterior(config) for name in arms}
    states: dict[str, dict] = {name: {} for name in arms}
    slice_budget = _slice_budget(total_budget, dim, config)

    def run_slice(name: str) -> None:
        before = rec.best
        ceiling = min(main_ceiling, rec.n + slice_budget)
        original = rec.budget
        rec.budget = ceiling
        try:
            _ARMS[name](ctx, states[name], ceiling - rec.n)
        except Exception as exc:  # noqa: BLE001
            if not _is_budget_exc(exc):
                raise
        finally:
            rec.budget = original
        threshold = config.improvement_atol + config.improvement_rtol * max(
            1.0, abs(before) if math.isfinite(before) else 1.0
        )
        posteriors[name].update(
            math.isfinite(rec.best) and (rec.best < before - threshold)
        )

    try:
        for name in arms:
            if rec.n + 4 > main_ceiling:
                break
            run_slice(name)
        while rec.n + 4 <= main_ceiling:
            if rng.random() < config.restart_floor:
                choice = RESTART_ARM
            else:
                draws = {k: posteriors[k].draw(rng) for k in arms}
                choice = max(draws, key=draws.get)
            run_slice(choice)
    except Exception as exc:  # noqa: BLE001
        if not _is_budget_exc(exc):
            raise

    if jac is not None and rec.n + 4 <= total_budget:
        try:
            _lbfgs(
                rec,
                jac,
                _incumbent(rec, low, high),
                bounds,
                total_budget - rec.n,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_budget_exc(exc):
                raise
    return _best_finite(counter.best)
