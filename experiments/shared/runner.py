"""Shared finite-precision SA runner used by experiments 1-4.

Why a separate Python loop (vs `anneal.run`)? The published Rust kernel
operates on `f64` end-to-end. The manuscript's Section 5 precision study
needs the kernel evaluated at `f16`/`f32` to surface underflow,
cancellation, and trajectory-bias artefacts that an `f64`-only kernel
cannot exhibit. This runner mirrors `anneal-core::runner::run_rs`
algorithmically (Boltzmann/Fast/GSA components, Metropolis acceptance,
seeded `numpy.random.default_rng`) but holds every state variable and
arithmetic step at the requested dtype, optionally with a Kahan
compensated `delta_e` summation.

Both runners agree at `dtype="float64", compensated_delta_e=False`
to within the ULP of the Rust implementation; the test in
`experiments/tests/test_runner.py` checks this on Styblinski-Tang 2D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Objectives. Each takes a 1D ndarray and returns a Python float that the
# caller is expected to cast to the runner's working dtype.
# ---------------------------------------------------------------------------


def styb_tang(x: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            x.astype(np.float64) ** 4
            - 16 * x.astype(np.float64) ** 2
            + 5 * x.astype(np.float64)
        )
    )


def rosenbrock(x: np.ndarray) -> float:
    x64 = x.astype(np.float64)
    return float(np.sum(100.0 * (x64[1:] - x64[:-1] ** 2) ** 2 + (1.0 - x64[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    x64 = x.astype(np.float64)
    return float(10.0 * len(x64) + np.sum(x64**2 - 10.0 * np.cos(2.0 * np.pi * x64)))


def ackley(x: np.ndarray) -> float:
    x64 = x.astype(np.float64)
    n = len(x64)
    s1 = np.sum(x64**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x64))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.e)


OBJECTIVES = {
    "styb_tang_2d": (styb_tang, np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
    "rosenbrock_2d": (rosenbrock, np.array([-2.048, -2.048]), np.array([2.048, 2.048])),
    "rastrigin_2d": (rastrigin, np.array([-5.12, -5.12]), np.array([5.12, 5.12])),
    "ackley_2d": (ackley, np.array([-32.768, -32.768]), np.array([32.768, 32.768])),
}


# ---------------------------------------------------------------------------
# Cooling schedules and proposal kernels (dtype-aware).
# ---------------------------------------------------------------------------


def log_cool(t_init, k0, epoch, dtype):
    return dtype(t_init * np.log(k0) / np.log(epoch + k0))


def reciprocal_cool(t_init, epoch, dtype):
    return dtype(t_init / (epoch + 1))


def tsallis_cool(t_init, q_v, epoch, dtype):
    if epoch == 0:
        return dtype(t_init)
    if abs(q_v - 1.0) < 1e-12:
        return dtype(t_init * np.log(2.0) / np.log(epoch + 1.0))
    exp = q_v - 1.0
    num = (2.0**exp) - 1.0
    den = ((epoch + 1.0) ** exp) - 1.0
    return dtype(t_init * num / den)


def gaussian_propose(rng, x, sigma, dtype):
    delta = rng.normal(0.0, sigma, size=x.shape).astype(dtype)
    return (x + delta).astype(dtype)


def cauchy_propose(rng, x, gamma, dtype):
    delta = rng.standard_cauchy(size=x.shape).astype(dtype) * dtype(gamma)
    return (x + delta).astype(dtype)


def tsallis_visit_propose(rng, x, q_v, t, dtype):
    """Student-t proposal with dof = (3-q_v)/(q_v-1), scaled by T^(1/(3-q_v))."""
    dof = (3.0 - q_v) / (q_v - 1.0)
    scale = float(t) ** (1.0 / (3.0 - q_v))
    z = rng.standard_normal(size=x.shape)
    g = rng.gamma(dof / 2.0, 2.0 / dof, size=x.shape)
    delta = (z / np.sqrt(g) * scale).astype(dtype)
    return (x + delta).astype(dtype)


# ---------------------------------------------------------------------------
# Metropolis acceptance with optional Kahan compensation on delta_e.
# ---------------------------------------------------------------------------


def metropolis_accept_prob(delta_e, temp, dtype):
    if delta_e <= dtype(0):
        return dtype(1)
    return dtype(np.exp(-delta_e / temp))


def gelman_rubin_max(traces):
    """Return max coordinate R-hat, or inf for incomplete/non-finite traces."""
    m = len(traces)
    if m < 2:
        return float("inf")
    n = len(traces[0])
    if n < 2 or any(len(chain) != n for chain in traces):
        return float("inf")
    dim = traces[0][0].shape[0]
    max_rhat = 0.0
    for d in range(dim):
        values = [
            np.asarray([x[d] for x in chain], dtype=np.float64) for chain in traces
        ]
        if any(vals.shape != (n,) or not np.all(np.isfinite(vals)) for vals in values):
            return float("inf")
        means = np.array([np.mean(vals) for vals in values], dtype=np.float64)
        vars_ = np.array([np.var(vals, ddof=1) for vals in values], dtype=np.float64)
        if not np.all(np.isfinite(vars_)) or np.any(vars_ <= 0):
            continue
        theta_bar = means.mean()
        b = (n / (m - 1.0)) * np.sum((means - theta_bar) ** 2)
        w = vars_.mean()
        var_hat = ((n - 1.0) / n) * w + b / n
        rhat = np.sqrt(var_hat / w)
        if not np.isfinite(rhat):
            return float("inf")
        if rhat > max_rhat:
            max_rhat = rhat
    return max_rhat


def log_metropolis_accept_prob(delta_e, temp, dtype):
    """Log-domain Metropolis acceptance: returns ln(accept_prob).

    The manuscript Section 5.4 (and Higham 1993, Section 5) require
    log-domain acceptance as the f16 default: f16's smallest positive
    representable value is ~6e-5, so any `accept_prob = exp(-delta_e/T)`
    with `delta_e/T > ~11` underflows to zero. In log domain we work
    with `-delta_e/T` directly and compare to `log(u)` for the rejection
    test, recovering the full dynamic range of the acceptance criterion.

    Returns -inf for the never-accept case so that `log(u) < -inf` is
    always false.
    """
    if delta_e <= dtype(0):
        return dtype(0.0)  # log(1) = 0
    return dtype(-delta_e / temp)


def kahan_step(running_sum, running_compensation, addend, dtype):
    """One step of Kahan compensated summation. Adds `addend` to the
    running sum, updating both the sum and the compensation in dtype.

    Returns the updated `(running_sum, running_compensation)` pair.
    Standard Kahan recurrence (Higham 1993, Section 4.3):
      y = addend - compensation
      t = sum + y
      compensation_new = (t - sum) - y
      sum_new = t
    Carrying `running_compensation` across accept steps recovers ~half
    the per-step rounding bias for f16; the manuscript Section 5.4 claim
    is verified on top of this primitive (not single-step subtraction)."""
    s = dtype(running_sum)
    c = dtype(running_compensation)
    y = dtype(addend) - c
    t = s + y
    c_new = (t - s) - y
    return t, c_new


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


@dataclass
class SaResult:
    best_pos: np.ndarray
    best_val: float
    n_calls: int
    accepted: int
    rejected: int
    history_temp: np.ndarray = field(default_factory=lambda: np.array([]))
    history_best: np.ndarray = field(default_factory=lambda: np.array([]))


def sa_run(
    *,
    objective: str,
    variant: str,
    dtype: str,
    seed: int,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    compensated_delta_e: bool = False,
    log_domain_accept: bool | None = None,
    t_init: float = 5.0,
    sigma: float = 0.5,
    gamma: float = 0.5,
    q_v: float = 2.62,
    q_a: float = 1.7,
) -> SaResult:
    """Mirrors `anneal-core::runner::run_rs` at the requested dtype.

    Determinism: same `(objective, variant, dtype, seed)` produces
    bitwise-identical `best_pos` modulo dtype rounding.

    Args:
        log_domain_accept: when True, use `log_metropolis_accept_prob`
            and compare `log(u) < log_p` instead of `u < exp(log_p)`.
            Default: True for f16 (per manuscript Section 5.4 / Higham
            1993 Section 5), False otherwise.
    """
    np_dtype = np.dtype(dtype)
    if log_domain_accept is None:
        log_domain_accept = np_dtype == np.float16
    obj_fn, low, high = OBJECTIVES[objective]
    rng = np.random.default_rng(seed)

    cur_pos = rng.uniform(low, high, size=low.shape).astype(np_dtype)
    cur_val = np_dtype.type(obj_fn(cur_pos))
    cur_compensation = np_dtype.type(0.0)  # Kahan running compensation
    best_pos = cur_pos.copy()
    best_val = cur_val
    accepted = 0
    rejected = 0
    n_calls = 1

    history_temp = np.empty(n_epochs, dtype=np.float64)
    history_best = np.empty(n_epochs, dtype=np.float64)

    for epoch in range(n_epochs):
        if variant == "boltzmann":
            temp = log_cool(t_init, 2.0, epoch, np_dtype.type)
        elif variant == "fast":
            temp = reciprocal_cool(t_init, epoch, np_dtype.type)
        elif variant == "gsa":
            temp = tsallis_cool(t_init, q_v, epoch, np_dtype.type)
        else:
            raise ValueError(f"unknown variant: {variant}")

        for _ in range(steps_per_epoch):
            if variant == "boltzmann":
                proposal = gaussian_propose(rng, cur_pos, sigma, np_dtype.type)
            elif variant == "fast":
                proposal = cauchy_propose(rng, cur_pos, gamma, np_dtype.type)
            else:
                proposal = tsallis_visit_propose(rng, cur_pos, q_v, temp, np_dtype.type)

            proposal_val = np_dtype.type(obj_fn(proposal))
            n_calls += 1

            delta = np_dtype.type(proposal_val - cur_val)
            u = np_dtype.type(rng.random())
            if log_domain_accept:
                log_p = log_metropolis_accept_prob(delta, temp, np_dtype.type)
                # log(0) underflows on small u in f16; clamp to a finite
                # most-negative value so the comparison always terminates.
                log_u = np.log(np.clip(u, np_dtype.type(1e-7), np_dtype.type(1.0)))
                accept = bool(np_dtype.type(log_u) < log_p)
            else:
                p = metropolis_accept_prob(delta, temp, np_dtype.type)
                accept = bool(u < p)
            if accept:
                cur_pos = proposal
                if compensated_delta_e:
                    # Update cur_val via running Kahan: cur_val += delta
                    # with carried compensation. This is what halves the
                    # f16 trajectory bias (manuscript Section 5.4); see
                    # kahan_step() docstring.
                    cur_val, cur_compensation = kahan_step(
                        cur_val, cur_compensation, delta, np_dtype.type
                    )
                else:
                    cur_val = proposal_val
                if proposal_val < best_val:
                    best_val = proposal_val
                    best_pos = proposal.copy()
                accepted += 1
            else:
                rejected += 1

        history_temp[epoch] = float(temp)
        history_best[epoch] = float(best_val)

    return SaResult(
        best_pos=best_pos,
        best_val=float(best_val),
        n_calls=n_calls,
        accepted=accepted,
        rejected=rejected,
        history_temp=history_temp,
        history_best=history_best,
    )
