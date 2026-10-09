"""Gradient-free ChemFit fitting with anneal (corrected review-response driver).

The snippet in the bug report had four defects:

1. ``fitter.evaluate`` / ``fitter.step`` do not exist. ChemFit's protocol
   is ``fitter.init()`` -> ``fitter.ask(params)`` per candidate ->
   ``fitter.tell()`` per step -> ``fitter.finish(best)``.
2. ``positions.reshape(...)`` without assignment is a no-op, so the
   reported loss never matched the evaluated geometry.
3. ``anneal.run`` took no initial parameters, so ChemFit's
   ``initial_params`` never reached the chain.
4. The classical presets ran on the unconstrained neighborhood, so nothing
   kept proposals inside ``[low, high]``. They now run reflected on the
   box; every evaluation is in-bounds.

This module keeps the ``run_anneal(benchmark_context)`` shape and fixes
all four by delegating to :func:`anneal.chemfit.fit_chemfit`.
"""

from typing import Any

from anneal.chemfit import fit_chemfit


def run_anneal(benchmark_context: dict[str, Any]) -> dict[str, Any]:
    """Fit ``benchmark_context["fitter"]`` within ``budget`` evaluations.

    The default ``method="portfolio"`` is the SOTA gradient-free driver:
    Thompson-allocated QMC restarts, basin hopping, differential evolution,
    GSA, parallel-tempering communicating chains, and trust-region polls.
    Pass ``method="boltzmann"`` for the single-chain ablation from the
    ChemFit initial parameters.
    """
    fitter = benchmark_context["fitter"]
    budget = int(benchmark_context["budget"])
    seed = int(benchmark_context.get("seed", 0))
    method = str(benchmark_context.get("method", "portfolio"))
    return fit_chemfit(fitter, budget=budget, method=method, seed=seed)


if __name__ == "__main__":  # smoke test without chemfit installed
    import numpy as np

    class _StubFitter:
        def __init__(self, initial_parameters, bounds=None):
            self.initial_parameters = initial_parameters
            self.bounds = bounds or {}

        def init(self):
            pass

        def ask(self, params):
            positions = np.asarray(params["positions"])
            return float(np.sum((positions - 1.0) ** 2))

        def tell(self, step=None):
            pass

        def finish(self, opt_params=None):
            return dict(opt_params)

    rng = np.random.default_rng(0)
    stub = _StubFitter({"positions": rng.uniform(-3, 3, size=(4, 3))})
    result = run_anneal({"fitter": stub, "budget": 800, "seed": 0})
    positions = np.asarray(result["positions"])
    assert positions.shape == (4, 3)
    assert np.all(positions >= -3.0 - 1e-9) and np.all(positions <= 3.0 + 1e-9)
    print("best loss:", float(np.sum((positions - 1.0) ** 2)))
