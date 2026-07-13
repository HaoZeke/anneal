"""Budget-matched SOTA comparison on the CUTEst set.

The ``hybrid_de`` entry is the anneal comparison point. It uses QMC seeding,
optional tensor/additive surrogate proposals, optional library GLE segments when
native gradients are available, and L-BFGS-B polish with counted gradients.
The SciPy baselines use the same objective/gradient budget accounting through
``Counter``.

Run on a host with pycutest bootstrapped (see bootstrap_cutest.sh):
  python -m experiments.scripts.sota_cutest --dim-cap 30 \
      --max-problems 60 --budget 8000 --seeds 3 --out results/sota_cutest.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    dual_annealing,
)

from experiments.anneal_sota import (
    DEFAULT_HYBRID_K_POLISH,
    DEFAULT_HYBRID_N_POLISH,
    qmc_annealed_hybrid,
)
from experiments.benchmarks.cutest_runner import (
    default_cutest_config,
    load,
)
from experiments.scripts.run_cutest_full_suite import (
    TargetProblem,
    list_target_problems,
)


class _Budget(Exception):
    pass


class Counter:
    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = budget
        self.n = 0
        self.objective_evals = 0
        self.grad_evals = 0
        self.best = float("inf")

    def _consume(self):
        if self.n >= self.budget:
            raise _Budget()
        self.n += 1

    def __call__(self, x):
        self._consume()
        self.objective_evals += 1
        v = float(self.fn(np.asarray(x, float).reshape(-1)))
        if math.isfinite(v) and v < self.best:
            self.best = v
        return v

    def record_initial(self, value):
        """Charge and retain the protocol's common starting objective."""
        self._consume()
        self.objective_evals += 1
        value = float(value)
        if math.isfinite(value) and value < self.best:
            self.best = value

    def counted_grad(self, grad):
        def jac(x):
            self._consume()
            self.grad_evals += 1
            return np.asarray(grad(np.asarray(x, float).reshape(-1)), float)

        return jac


def _counted_jac(counter, grad):
    return counter.counted_grad(grad) if grad else None


def _auto_sigma(low, high, dim):
    diag = float(np.linalg.norm(high - low))
    return float(np.clip(0.25 * diag / dim, 1e-6, diag / np.sqrt(dim)))


def _comparison_box(prob):
    low = np.asarray(getattr(prob, "design_low", prob.low), dtype=np.float64)
    high = np.asarray(getattr(prob, "design_high", prob.high), dtype=np.float64)
    anchor = getattr(prob, "x0", None)
    if anchor is None:
        anchor = 0.5 * (low + high)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor.shape != low.shape:
        anchor = 0.5 * (low + high)
    return low, high, np.clip(anchor, low, high)


def classical(counter, low, high, dim, grad, rng, anchor=None):
    sigma = _auto_sigma(low, high, dim)
    x = (
        np.asarray(anchor, dtype=np.float64).copy()
        if anchor is not None
        else rng.uniform(low, high)
    )
    fx = counter(x)
    epoch = 0
    try:
        while True:
            temp = 5.0 * np.log(2.0) / np.log(epoch + 2.0)
            for _ in range(150):
                y = np.clip(x + rng.normal(0.0, sigma, dim), low, high)
                fy = counter(y)
                if fy < fx or rng.random() < np.exp(-(fy - fx) / max(temp, 1e-12)):
                    x, fx = y, fy
            epoch += 1
    except _Budget:
        pass
    return counter.best


def hybrid_de(
    counter,
    low,
    high,
    dim,
    grad,
    rng,
    n_polish=DEFAULT_HYBRID_N_POLISH,
    k_polish=DEFAULT_HYBRID_K_POLISH,
    config=None,
    anchor=None,
):
    """Anneal benchmark entry backed by ``qmc_annealed_hybrid``."""
    return qmc_annealed_hybrid(
        counter,
        low,
        high,
        dim,
        grad,
        rng,
        n_polish=n_polish,
        k_polish=k_polish,
        config=config,
        anchor=anchor,
    )


def sci_basinhopping(counter, low, high, dim, grad, rng, anchor=None):
    bounds = list(zip(low, high))
    x0 = (
        np.asarray(anchor, dtype=np.float64).copy()
        if anchor is not None
        else rng.uniform(low, high)
    )
    jac = _counted_jac(counter, grad)
    mk = {"method": "L-BFGS-B", "bounds": bounds}
    if jac is not None:
        mk["jac"] = jac
    try:
        basinhopping(
            counter,
            x0,
            niter=10**6,
            minimizer_kwargs=mk,
            seed=int(rng.integers(1 << 31)),
        )
    except _Budget:
        pass
    return counter.best


def portfolio(counter, low, high, dim, grad, rng, anchor=None, policy="auto"):
    """Native Thompson-allocated portfolio over anneal building blocks."""
    del anchor
    import anneal

    remaining = counter.budget - counter.n
    if remaining <= 0:
        return counter.best
    jac = counter.counted_grad(grad) if grad is not None else None
    try:
        out = anneal.global_optimize(
            counter,
            low,
            high,
            budget=remaining,
            seed=int(rng.integers(1 << 31)),
            grad_fn=jac,
            policy=policy,
        )
        best = float(out.get("best_val", float("inf")))
        pos = np.asarray(out.get("best_pos", []), dtype=float).reshape(-1)
        # Feasibility gate: OOB bests are non-solutions (score as +inf).
        if pos.size == low.size:
            if np.any(pos < low - 1e-8) or np.any(pos > high + 1e-8):
                best = float("inf")
        if math.isfinite(best) and best < counter.best:
            counter.best = best
    except _Budget:
        pass
    return counter.best


def portfolio_legacy(counter, low, high, dim, grad, rng, anchor=None):
    """Pre-regime portfolio (flat order, Beta(1,1)) for same-protocol A/B."""
    return portfolio(counter, low, high, dim, grad, rng, anchor=anchor, policy="legacy")


def dmc_pop(counter, low, high, dim, grad, rng, anchor=None):
    """Population-controlled diffusion arm under the shared work-unit budget."""
    del dim
    import anneal

    remaining = counter.budget - counter.n
    if remaining <= 0:
        return counter.best
    jac = counter.counted_grad(grad) if grad is not None else None
    try:
        out = anneal.dmc_population_optimize(
            counter,
            low,
            high,
            budget=remaining,
            seed=int(rng.integers(1 << 31)),
            grad_fn=jac,
            target_n=min(24, max(4, remaining // 16)),
            steps_per_control=3,
            x0=anchor,
        )
        best = float(out.get("best_val", float("inf")))
        pos = np.asarray(out.get("best_pos", []), dtype=float).reshape(-1)
        if pos.size == low.size:
            if np.any(pos < low - 1e-8) or np.any(pos > high + 1e-8):
                best = float("inf")
        if math.isfinite(best) and best < counter.best:
            counter.best = best
    except _Budget:
        pass
    return counter.best


def sci_dual_annealing(counter, low, high, dim, grad, rng, anchor=None):
    bounds = list(zip(low, high))
    x0 = np.asarray(anchor, dtype=np.float64).copy() if anchor is not None else None
    try:
        dual_annealing(
            counter,
            bounds,
            maxfun=10**9,
            maxiter=10**9,
            seed=int(rng.integers(1 << 31)),
            x0=x0,
        )
    except _Budget:
        pass
    return counter.best


def sci_de(counter, low, high, dim, grad, rng, anchor=None):
    del anchor
    bounds = list(zip(low, high))
    try:
        differential_evolution(
            counter,
            bounds,
            maxiter=10**6,
            polish=True,
            seed=int(rng.integers(1 << 31)),
            tol=0,
        )
    except _Budget:
        pass
    return counter.best


def cma_es(counter, low, high, dim, grad, rng, anchor=None):
    """CMA-ES restarts (pycma) under the shared budget counter."""
    import cma

    width = np.where(high > low, high - low, 1.0)
    try:
        while counter.n < counter.budget:
            before = counter.n
            x0 = (
                np.asarray(anchor, dtype=np.float64).copy()
                if anchor is not None and counter.n == 0
                else rng.uniform(low, high)
            )
            try:
                es = cma.CMAEvolutionStrategy(
                    x0,
                    0.25 * float(np.mean(width)),
                    {
                        "bounds": [list(low), list(high)],
                        "verbose": -9,
                        "seed": int(rng.integers(1 << 31)),
                        "maxfevals": counter.budget - counter.n,
                    },
                )
                while not es.stop() and counter.n < counter.budget:
                    xs = es.ask()
                    es.tell(xs, [counter(x) for x in xs])
            except ValueError as exc:
                if not _is_restartable_cma_error(exc) or counter.n == before:
                    raise
    except _Budget:
        pass
    return counter.best


def _is_restartable_cma_error(exc):
    return isinstance(exc, ValueError) and str(exc) == (
        "not yet initialized (dimension needed)"
    )


def ngopt(counter, low, high, dim, grad, rng, anchor=None):
    """Nevergrad NGOpt wizard, restarted until the shared budget is spent.

    NGOpt can terminate before its declared budget; restart-until-budget
    matches the other baselines. Wall-clock warning: NGOpt's per-ask
    overhead dwarfs cheap CUTEst objectives, so full-matrix runs are
    disclosed with their wall cost in the campaign notes.
    """
    import nevergrad as ng

    del grad, anchor
    try:
        while counter.n < counter.budget:
            before = counter.n
            param = ng.p.Array(
                init=rng.uniform(low, high).astype(np.float64)
            ).set_bounds(list(low), list(high))
            param.random_state = np.random.RandomState(int(rng.integers(1 << 31)))
            opt = ng.optimizers.NGOpt(
                parametrization=param, budget=counter.budget - counter.n
            )
            opt.minimize(lambda x: counter(np.asarray(x, dtype=np.float64)))
            if counter.n == before:
                break
    except _Budget:
        pass
    return counter.best


def bobyqa(counter, low, high, dim, grad, rng, anchor=None):
    """Py-BOBYQA multistart-restarts under the shared budget counter."""
    import pybobyqa

    del grad
    try:
        while counter.n < counter.budget:
            x0 = (
                np.asarray(anchor, dtype=np.float64).copy()
                if anchor is not None and counter.n == 0
                else rng.uniform(low, high)
            )
            remaining = counter.budget - counter.n
            if remaining < 2 * dim + 2:
                break
            pybobyqa.solve(
                counter,
                x0,
                bounds=(np.asarray(low), np.asarray(high)),
                maxfun=remaining,
                seek_global_minimum=True,
                scaling_within_bounds=True,
                do_logging=False,
            )
    except _Budget:
        pass
    return counter.best


@dataclass
class _TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, self.dim / self.batch_size)
        )


def _turbo_update(state, y_next):
    candidate = float(y_next.max())
    threshold = 1e-3 * max(1.0, abs(state.best_value))
    if candidate > state.best_value + threshold:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, candidate)
    state.restart_triggered = state.length < state.length_min
    return state


def _fit_turbo_or_restart(mll, fitter, model_fitting_error):
    try:
        fitter(mll)
    except model_fitting_error:
        return False
    return True


def _finite_turbo_targets(values, *, reference):
    """Replace invalid maximization targets with a finite pessimistic value."""

    targets = np.asarray(values, dtype=np.float64).copy()
    reference = np.asarray(reference, dtype=np.float64)
    finite = np.concatenate(
        (targets[np.isfinite(targets)], reference[np.isfinite(reference)])
    )
    if finite.size == 0:
        raise ValueError("TuRBO requires at least one finite objective observation")
    scale = max(1.0, float(np.max(np.abs(finite))))
    scaled = finite / scale
    penalty = max(1.0 / scale, float(np.max(scaled) - np.min(scaled)))
    fallback = (float(np.min(scaled)) - penalty) * scale
    if not math.isfinite(fallback):
        fallback = -np.finfo(np.float64).max
    targets[~np.isfinite(targets)] = fallback
    return targets


def turbo(counter, low, high, dim, grad, rng, anchor=None):
    """BoTorch TuRBO-1 with Thompson batches under the shared work cap.

    BoTorch's default fit retries handle unstable hyperparameter fits. An
    exhausted fit retry starts a fresh trust region while preserving the
    incumbent and charging every initial-design evaluation.
    """
    del grad
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.exceptions.errors import InputDataError, ModelFittingError
    from botorch.generation import MaxPosteriorSampling
    from botorch.models import SingleTaskGP
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dtype = torch.double
    torch.manual_seed(int(rng.integers(1 << 31)))
    width = np.asarray(high - low, dtype=np.float64)
    anchor = (
        np.asarray(anchor, dtype=np.float64).reshape(-1)
        if anchor is not None
        else 0.5 * (low + high)
    )
    anchor_unit = np.clip((anchor - low) / width, 0.0, 1.0)
    x_anchor = torch.tensor(anchor_unit, dtype=dtype).unsqueeze(0)
    y_anchor = torch.tensor([[-counter.best]], dtype=dtype)

    def evaluate(unit_points, reference):
        values = []
        for point in unit_points.detach().cpu().numpy():
            values.append(-counter(low + width * point))
        finite = _finite_turbo_targets(
            np.asarray(values, dtype=np.float64).reshape(-1, 1),
            reference=reference.detach().cpu().numpy(),
        )
        return torch.tensor(finite, dtype=dtype)

    while counter.n < counter.budget:
        remaining = counter.budget - counter.n
        n_init = min(remaining, max(7, 2 * dim))
        sobol = torch.quasirandom.SobolEngine(
            dim, scramble=True, seed=int(rng.integers(1 << 31))
        )
        x_initial = sobol.draw(n_init).to(dtype=dtype)
        y_initial = evaluate(x_initial, y_anchor)
        x_data = torch.cat((x_anchor, x_initial), dim=0)
        y_data = torch.cat((y_anchor, y_initial), dim=0)
        state = _TurboState(
            dim=dim,
            batch_size=min(4, max(1, counter.budget - counter.n)),
            best_value=float(y_data.max()),
        )
        while counter.n < counter.budget and not state.restart_triggered:
            y_scale = y_data.abs().max().clamp_min(1.0)
            scaled_y = y_data / y_scale
            y_std = scaled_y.std(unbiased=False).clamp_min(1e-12)
            train_y = (scaled_y - scaled_y.mean()) / y_std
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covariance = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                )
            )
            try:
                model = SingleTaskGP(
                    x_data,
                    train_y,
                    covar_module=covariance,
                    likelihood=likelihood,
                )
            except InputDataError:
                state.restart_triggered = True
                break
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not _fit_turbo_or_restart(mll, fit_gpytorch_mll, ModelFittingError):
                    state.restart_triggered = True
                    break

            center = x_data[train_y.argmax(), :].clone()
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / dim))
            lower = torch.clamp(center - weights * state.length / 2.0, 0.0, 1.0)
            upper = torch.clamp(center + weights * state.length / 2.0, 0.0, 1.0)
            n_candidates = min(5000, max(512, 100 * dim))
            candidates = sobol.draw(n_candidates).to(dtype=dtype)
            candidates = lower + (upper - lower) * candidates
            probability = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype) <= probability
            empty = torch.where(mask.sum(dim=1) == 0)[0]
            if len(empty):
                mask[empty, torch.randint(dim, size=(len(empty),))] = True
            candidate_set = center.expand(n_candidates, dim).clone()
            candidate_set[mask] = candidates[mask]
            batch_size = min(state.batch_size, counter.budget - counter.n)
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    x_next = sampler(candidate_set, num_samples=batch_size)
            y_next = evaluate(x_next, y_data)
            state = _turbo_update(state, y_next)
            x_data = torch.cat((x_data, x_next), dim=0)
            y_data = torch.cat((y_data, y_next), dim=0)
    return counter.best


def cma_es_ipop(counter, low, high, dim, grad, rng, anchor=None):
    """IPOP-style CMA-ES: restart with growing population (stronger baseline)."""
    import cma

    del grad
    width = np.where(high > low, high - low, 1.0)
    mean_w = float(np.mean(width))
    pop0 = int(np.clip(4 + 3 * np.log(max(dim, 1)), 8, 40))
    pop = pop0
    try:
        while counter.n < counter.budget:
            before = counter.n
            x0 = (
                np.asarray(anchor, dtype=np.float64).copy()
                if anchor is not None and counter.n == 0
                else rng.uniform(low, high)
            )
            remaining = counter.budget - counter.n
            try:
                es = cma.CMAEvolutionStrategy(
                    x0,
                    0.3 * mean_w,
                    {
                        "bounds": [list(low), list(high)],
                        "verbose": -9,
                        "seed": int(rng.integers(1 << 31)),
                        "maxfevals": remaining,
                        "popsize": pop,
                    },
                )
                while not es.stop() and counter.n < counter.budget:
                    xs = es.ask()
                    es.tell(xs, [counter(x) for x in xs])
            except ValueError as exc:
                if not _is_restartable_cma_error(exc) or counter.n == before:
                    raise
            # IPOP: grow population after each restart until budget ends.
            pop = min(pop * 2, 200)
    except _Budget:
        pass
    return counter.best


METHODS = {
    "portfolio": portfolio,
    "portfolio_legacy": portfolio_legacy,
    "dmc_pop": dmc_pop,
    "hybrid_de": hybrid_de,
    "basinhopping": sci_basinhopping,
    "dual_annealing": sci_dual_annealing,
    "diff_evol": sci_de,
    "cma_es": cma_es,
    "cma_es_ipop": cma_es_ipop,
    "ngopt": ngopt,
    "bobyqa": bobyqa,
    "turbo": turbo,
    "classical": classical,
}
FIELDNAMES = [
    "problem",
    "dim",
    "method",
    "seed",
    "initial",
    "best",
    "evals",
    "objective_evals",
    "grad_evals",
    "status",
]


def _write_sota_row(writer, stream, row):
    writer.writerow(row)
    stream.flush()


def _campaign_exit_code(rows):
    successful = {"ok", "budget_exhausted"}
    return int(any(row["status"] not in successful for row in rows))


def run_method_cell(
    *,
    method_name,
    method,
    problem,
    dim,
    seed,
    initial,
    counter,
    low,
    high,
    grad,
    anchor,
):
    """Run one solver cell and return a status-bearing accounting row."""
    rng = np.random.default_rng(seed)
    status = "ok"
    try:
        counter.record_initial(initial)
        best = method(counter, low, high, dim, grad, rng, anchor=anchor)
    except _Budget:
        best = counter.best
        status = "budget_exhausted"
    except Exception as exc:  # noqa: BLE001
        best = float("inf")
        status = f"error:{type(exc).__name__}"
    if not math.isfinite(float(best)) and status == "ok":
        status = "nonfinite"
    if status.startswith("error:") or status == "nonfinite":
        best = float("inf")
    return {
        "problem": problem,
        "dim": dim,
        "method": method_name,
        "seed": seed,
        "initial": initial,
        "best": best,
        "evals": counter.n,
        "objective_evals": counter.objective_evals,
        "grad_evals": counter.grad_evals,
        "status": status,
    }


def _shard_targets(targets, shard_index: int, shard_count: int):
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [
        target
        for index, target in enumerate(targets)
        if index % shard_count == shard_index
    ]


def load_problem_manifest(path):
    """Load the committed CSV population used by a benchmark campaign."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"problem", "kind", "dim"}
    if not rows:
        raise ValueError("problem manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"problem manifest missing columns: {sorted(missing)}")
    targets = []
    seen = set()
    for row in rows:
        name = row["problem"].strip()
        kind = row["kind"].strip()
        dim = int(row["dim"])
        if not name or kind not in {"unconstrained", "bound"} or dim <= 0:
            raise ValueError(f"invalid problem manifest row: {row}")
        if name in seen:
            raise ValueError(f"duplicate problem in manifest: {name}")
        seen.add(name)
        targets.append(TargetProblem(name=name, kind=kind, dim=dim))
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/sota_cutest.csv")
    p.add_argument("--dim-cap", type=int, default=30)
    p.add_argument("--max-problems", type=int, default=60)
    p.add_argument("--budget", type=int, default=8000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument(
        "--bench-root",
        default=None,
        help="Project root containing .bench/ with CUTEst, SIFDecode, and sif.",
    )
    p.add_argument(
        "--pycutest-cache",
        default=None,
        help="Explicit PyCUTEst cache directory; defaults to .bench/cache.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Stable shard index for distributed CUTEst sweeps.",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of stable shards in the distributed CUTEst sweep.",
    )
    p.add_argument(
        "--methods", default=None, help="Comma-separated subset of methods to run."
    )
    p.add_argument(
        "--problems-file",
        default=None,
        help="Optional newline-separated problem names (paper list). "
        "When set, only these names are run (still filtered by dim-cap).",
    )
    p.add_argument(
        "--problem-manifest",
        default=None,
        help="Committed CSV with problem, kind, and dimension columns.",
    )
    p.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit non-zero if any cell status is not ok/budget_exhausted. "
        "Default is to exit 0 after writing the status-bearing matrix so "
        "campaign harnesses keep failed cells as non-wins.",
    )
    args = p.parse_args()
    if args.problem_manifest and args.problems_file:
        p.error("use only one of --problem-manifest and --problems-file")
    if args.methods:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = sorted(set(requested) - set(METHODS))
        if unknown:
            p.error(f"unknown methods: {unknown}")
        methods = {m: METHODS[m] for m in requested}
    else:
        methods = dict(METHODS)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    config = default_cutest_config(args.bench_root, cache_dir=args.pycutest_cache)

    if args.problem_manifest:
        all_targets = [
            target
            for target in load_problem_manifest(args.problem_manifest)
            if target.dim <= args.dim_cap
        ][: args.max_problems]
    elif args.problems_file:
        names = [
            line.strip()
            for line in open(args.problems_file, encoding="utf-8")
            if line.strip() and not line.strip().startswith("#")
        ]
        # Prefer paper order; load will skip failures later.
        # kind/dim filled after load; placeholder for listing only.
        all_targets = [
            TargetProblem(name=n, kind="unconstrained", dim=0) for n in names
        ]
        all_targets = all_targets[: args.max_problems]
    else:
        all_targets = list_target_problems(args.dim_cap, config=config)[
            : args.max_problems
        ]
    targets = _shard_targets(all_targets, args.shard_index, args.shard_count)
    print(
        f"{len(targets)} CUTEst problems, dim <= {args.dim_cap}, "
        f"budget {args.budget}, shard {args.shard_index}/{args.shard_count}",
        flush=True,
    )
    rows = []
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        f.flush()
        for t in targets:
            try:
                prob = load(t.name, sif_params=None, config=config)
            except Exception as exc:  # noqa: BLE001
                status = f"load_error:{type(exc).__name__}"
                for s in range(args.seeds):
                    for name in methods:
                        row = {
                            "problem": t.name,
                            "dim": t.dim,
                            "method": name,
                            "seed": s,
                            "initial": float("inf"),
                            "best": float("inf"),
                            "evals": 0,
                            "objective_evals": 0,
                            "grad_evals": 0,
                            "status": status,
                        }
                        rows.append(row)
                        _write_sota_row(
                            w,
                            f,
                            row,
                        )
                print(f"  {t.name}: {status}", flush=True)
                continue
            low, high, anchor = _comparison_box(prob)
            dim = prob.dim
            initial = float(prob.fn(anchor))
            if not math.isfinite(initial):
                raise ValueError(f"non-finite starting objective for {t.name}")
            for s in range(args.seeds):
                for name, fnc in methods.items():
                    c = Counter(prob.fn, args.budget)
                    row = run_method_cell(
                        method_name=name,
                        method=fnc,
                        problem=t.name,
                        dim=dim,
                        seed=s,
                        initial=initial,
                        counter=c,
                        low=low,
                        high=high,
                        grad=prob.grad,
                        anchor=anchor,
                    )
                    rows.append(row)
                    _write_sota_row(w, f, row)
            print(f"  {t.name} (dim {dim}) done", flush=True)

    # win-rate summary: per (problem, seed), which method reached the lowest best
    wins = {m: 0 for m in methods}
    cells = {}
    for r in rows:
        cells.setdefault((r["problem"], r["seed"]), []).append((r["method"], r["best"]))
    for cand in cells.values():
        finite = [(m, v) for m, v in cand if math.isfinite(v)]
        if finite:
            bv = min(v for _, v in finite)
            for m, v in finite:
                if v <= bv + 1e-9:
                    wins[m] += 1
    print(f"\nWin counts over {len(cells)} cells:")
    for m in sorted(methods, key=lambda k: -wins[k]):
        print(f"  {m:>14} {wins[m]:5d}")
    print(f"Wrote {len(rows)} rows to {args.out}")
    exit_code = _campaign_exit_code(rows)
    if exit_code:
        failures = sum(row["status"] not in {"ok", "budget_exhausted"} for row in rows)
        print(f"Campaign contains {failures} unsuccessful cells", file=sys.stderr)
        if not args.strict_exit:
            # Protocol: unsuccessful cells are ranked as non-wins, not harness
            # aborts. Snakemake deletes outputs on non-zero exit, which would
            # discard a complete status-bearing shard.
            print(
                "Non-strict exit: keeping status-bearing CSV (pass --strict-exit to fail)",
                file=sys.stderr,
            )
            return 0
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
